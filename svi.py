import geopandas as gpd
from urbanworm import GeoTaggedData
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--key", type=string)
args = parser.parse_args()

def filter_area(data, minm=0, maxm=None):
    utm = data.estimate_utm_crs()
    data = data.to_crs(utm)
    data["footprint_area"] = data.geometry.area
    data = data[data["footprint_area"] >= float(minm)]
    if maxm is not None:
        data = data[data["footprint_area"] < float(maxm)]
    return data.to_crs(epsg=4326)

def intersect_geo(geo1, geo2):
    utm = geo1.estimate_utm_crs()
    geo1_m = geo1.to_crs(utm)
    geo2_m = geo2.to_crs(utm)

    geo1_cent = geo1_m.copy()
    geo1_cent["centroid"] = geo1_cent.geometry.centroid
    geo1_pts = geo1_cent.set_geometry("centroid")

    joined = gpd.sjoin(geo1_pts, geo2_m, how="inner", predicate="intersects")
    out = geo1.loc[joined.index.unique()].copy()
    return out

buildings_detroit = gpd.read_file("data/buildings.geojson").to_crs(4326)
zoning = gpd.read_file("data/zoning.geojson").to_crs(4326)
buildings_detroit.geometry = buildings_detroit.force_2d()
residential = zoning[(zoning['ZONING_REV'] == 'R1') |
                     (zoning['ZONING_REV'] == 'R2') |
                     (zoning['ZONING_REV'] == 'R3') |
                     (zoning['ZONING_REV'] == 'R4') |
                     (zoning['ZONING_REV'] == 'R5') |
                     (zoning['ZONING_REV'] == 'R6')
                    ]

residential_buildings = intersect_geo(buildings_detroit, residential)
residential_buildings = filter_area(residential_buildings, minm=60, maxm=200)
residential_buildings = gpd.sjoin(residential_buildings, residential, how="inner", predicate="intersects")
residential_buildings = residential_buildings[residential_buildings['parcel_id'] != 'CONDO BUILDING']
print(f"Total residential buildings: {len(residential_buildings)}")
print(f"processing from index {args.start} to {args.end}")
residential_buildings = residential_buildings.iloc[args.start:args.end]

gtd = GeoTaggedData(units = residential_buildings)
gtd.get_svi_from_locations(key=args.key, 
                           id_column="parcel_id",
                           distance=30,
                           fov=45,
                           multi_num=3,
                           reoriented=True,
                           interval=2,
                           height=400,
                           width=300,
                           year=(2024,2025),
                           silent=True
                           )

gtd.download_to_dir(data='svi', to_dir='svi_computed')
