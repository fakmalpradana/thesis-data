"""OSM drainage ways -> networkx graph, stations snapped to nodes, shortest paths.

    python3 scripts/canal_graph.py

Outputs (tracked, small):
    reference/jakarta/canal_graph.gpkg     - edges as LineStrings (waterway, name, length_m, z_mean)
    reference/jakarta/canal_graph.graphml  - same graph for networkx re-use
    reference/jakarta/station_graph_distance.csv - pairwise shortest path (km) between
        stations in the same connected component
"""
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent.parent
OSM_JSON = ROOT / "forcing-acquisition/data/raw/osm_drainage/osm_drainage_106.6_-6.5_107.1_-6.0.json"
DTM10 = ROOT / "forcing-acquisition/data/static/dem/jakarta/grid10m/dtm_10m.tif"
OUT_DIR = ROOT / "reference/jakarta"
WATERWAYS = {"river", "canal", "drain", "stream", "ditch"}
SNAP_MAX_M = 300.0
SEA_OUTLET_IDS = {140, 162, 170}  # Ancol (Laut/Kali/Flushing) / Pasar Ikan

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True).transform


def load_graph() -> tuple[nx.Graph, dict]:
    data = json.loads(OSM_JSON.read_text())
    ways = [e for e in data["elements"] if e["type"] == "way"
            and e.get("tags", {}).get("waterway") in WATERWAYS]

    g = nx.Graph()
    node_xy: dict[int, tuple[float, float]] = {}
    for w in ways:
        nodes = w["nodes"]
        geom = w["geometry"]
        for nid, pt in zip(nodes, geom):
            if nid not in node_xy:
                x, y = to_utm(pt["lon"], pt["lat"])
                node_xy[nid] = (x, y)
                g.add_node(nid, x=x, y=y, lon=pt["lon"], lat=pt["lat"])
        tags = w.get("tags", {})
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a == b:
                continue
            (x1, y1), (x2, y2) = node_xy[a], node_xy[b]
            length_m = float(np.hypot(x2 - x1, y2 - y1))
            if g.has_edge(a, b):
                continue  # ponytail: keep first duplicate edge, not worth merging attrs
            g.add_edge(a, b, length_m=length_m, waterway=tags.get("waterway", ""),
                       name=tags.get("name", ""))
    return g, node_xy


def load_stations() -> gpd.GeoDataFrame:
    con = duckdb.connect(str(ROOT / "catalog.duckdb"), read_only=True)
    df = con.execute("select stasiun_id, nama, lat, lon from stations "
                      "where stasiun_id in (140,166,162,167,150,126,164,169,107,179,181,184,187,170)").df()
    row140 = df.loc[df.stasiun_id == 140].iloc[0]
    # 170 has no coords: place ~200 m east of 140 (plan says assumed, output-only flag)
    dlon = 200.0 / (111320.0 * np.cos(np.radians(row140.lat)))
    df.loc[df.stasiun_id == 170, ["lat", "lon"]] = [row140.lat, row140.lon + dlon]
    df["koordinat_sumber"] = np.where(df.stasiun_id == 170, "assumed near 140", "stations table")
    x, y = to_utm(df.lon.values, df.lat.values)
    df["x"], df["y"] = x, y
    return df


def snap_stations(df, node_xy: dict) -> tuple[dict, list]:
    ids = list(node_xy)
    coords = np.array([node_xy[i] for i in ids])
    tree = cKDTree(coords)
    dist, idx = tree.query(df[["x", "y"]].values, k=1)
    snapped, unsnapped = {}, []
    for i, row in enumerate(df.itertuples()):
        d = float(dist[i])
        if d <= SNAP_MAX_M:
            snapped[row.stasiun_id] = (ids[idx[i]], d)
        else:
            unsnapped.append((row.stasiun_id, row.nama, d))
    return snapped, unsnapped


def sample_edge_z(g: nx.Graph):
    with rasterio.open(DTM10) as src:
        arr = src.read(1)
        inv = ~src.transform
        for u, v, data in g.edges(data=True):
            xs = np.linspace(g.nodes[u]["x"], g.nodes[v]["x"], 3)
            ys = np.linspace(g.nodes[u]["y"], g.nodes[v]["y"], 3)
            zs = []
            for x, y in zip(xs, ys):
                col, row = inv * (x, y)
                col, row = int(col), int(row)
                if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                    z = arr[row, col]
                    if not np.isnan(z):
                        zs.append(z)
            data["z_mean"] = float(np.mean(zs)) if zs else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g, node_xy = load_graph()
    stations = load_stations()
    snapped, unsnapped = snap_stations(stations, node_xy)
    sample_edge_z(g)

    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("\n-- snap table (station -> node, distance m) --")
    id2name = dict(zip(stations.stasiun_id, stations.nama))
    for sid, (nid, d) in snapped.items():
        print(f"{sid} {id2name[sid]:<30} -> node {nid}  {d:.1f} m")
    if unsnapped:
        print("-- unsnapped (> 300 m) --")
        for sid, nama, d in unsnapped:
            print(f"{sid} {nama:<30} {d:.1f} m")

    # component per station
    comp_of = {n: i for i, comp in enumerate(nx.connected_components(g)) for n in comp}
    station_comp = {sid: comp_of[nid] for sid, (nid, _) in snapped.items()}
    print("\n-- component summary --")
    from collections import Counter
    for comp_id, cnt in Counter(station_comp.values()).items():
        members = [sid for sid, c in station_comp.items() if c == comp_id]
        print(f"component {comp_id}: {cnt} station(s) -> {members}")

    sea_comps = {station_comp[s] for s in SEA_OUTLET_IDS if s in station_comp}
    sea_connected = [sid for sid, c in station_comp.items() if c in sea_comps]
    print(f"\nstations hydraulically connected to sea outlets {sorted(SEA_OUTLET_IDS)}: {sea_connected}")

    # pairwise shortest path within same component
    rows = []
    sids = sorted(snapped)
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            if station_comp[a] != station_comp[b]:
                continue
            na, nb = snapped[a][0], snapped[b][0]
            try:
                d_m = nx.shortest_path_length(g, na, nb, weight="length_m")
            except nx.NetworkXNoPath:
                continue
            rows.append({"station_a": a, "station_b": b, "distance_km": round(d_m / 1000, 3)})
    import pandas as pd
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(OUT_DIR / "station_graph_distance.csv", index=False)
    print(f"\n-- shortest-path table ({len(dist_df)} pairs) --")
    print(dist_df.to_string(index=False))

    below_msl = [(u, v, d["z_mean"]) for u, v, d in g.edges(data=True) if d["z_mean"] < 0]
    print(f"\nedges below MSL (z_mean < 0): {len(below_msl)} / {g.number_of_edges()}")

    # write graphml (drop nothing; graphml handles str/float/int attrs)
    nx.write_graphml(g, OUT_DIR / "canal_graph.graphml")

    # write gpkg: one LineString per edge
    edge_rows = []
    for u, v, d in g.edges(data=True):
        edge_rows.append({
            "u": u, "v": v, "waterway": d["waterway"], "name": d["name"],
            "length_m": d["length_m"], "z_mean": d["z_mean"],
            "geometry": LineString([(g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])]),
        })
    gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs="EPSG:32748")
    gdf.to_file(OUT_DIR / "canal_graph.gpkg", driver="GPKG")

    assert g.number_of_nodes() > 0 and any(cnt >= 3 for cnt in Counter(station_comp.values()).values()), \
        "no component has >= 3 stations - graph or snap likely broken"


if __name__ == "__main__":
    main()
