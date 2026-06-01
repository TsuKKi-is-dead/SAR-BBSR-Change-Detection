from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

def get_access_token(u, p):
    r = requests.post(cfg.CDSE_AUTH_URL, data={"grant_type":"password","username":u,"password":p,"client_id":"cdse-public"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def search_products(limit=5):
    ds = "2024-02-01T00:00:00.000Z"
    de = "2024-02-29T23:59:59.000Z"
    filt = (f"Collection/Name eq 'SENTINEL-1' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{cfg.PRODUCT_TYPE}') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' and att/OData.CSC.StringAttribute/Value eq '{cfg.SENSOR_MODE}') and ContentDate/Start gt {ds} and ContentDate/Start lt {de} and OData.CSC.Intersects(area=geography'SRID=4326;{cfg.AOI_WKT}')")
    print(f"\nQuerying CDSE for Sentinel-1 {cfg.PRODUCT_TYPE} over {cfg.AOI_NAME}")
    print(f"  Date range : 2024-01-01 → 2024-01-31")
    r = requests.get(cfg.CDSE_CATALOG_URL, params={"$filter":filt,"$orderby":"ContentDate/Start desc","$top":limit,"$expand":"Attributes"}, timeout=60)
    r.raise_for_status()
    products = r.json().get("value", [])
    print(f"  Found      : {len(products)} product(s)\n")
    return products

def download_product(product, token):
    pid, name = product["Id"], product["Name"]
    dest = cfg.DATA_RAW / f"{name}.zip"
    if dest.exists():
        print(f"  [skip] already exists: {dest.name}"); return dest
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    url = f"{cfg.CDSE_DOWNLOAD_BASE}({pid})/$value"
    print(f"  Downloading {name}\n  → {dest}")
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {done/total*100:5.1f}%  {done>>20}/{total>>20} MB", end="", flush=True)
    print(); return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--list",  action="store_true")
    args = parser.parse_args()

    username = os.getenv("CDSE_USER")
    password = os.getenv("CDSE_PASS")

    print(f"DEBUG .env loaded — CDSE_USER: {username}")

    products = search_products(limit=args.limit)
    if not products:
        print("No products found."); sys.exit(0)

    for i, p in enumerate(products):
        print(f"  [{i+1}] {p['Name']}")
        print(f"       Date : {p['ContentDate']['Start'][:10]}   Size : {int(p.get('ContentLength',0))>>20} MB")

    if args.list:
        print("\n--list flag set — skipping download."); sys.exit(0)

    if not username or not password:
        print("\nERROR: CDSE_USER / CDSE_PASS not found in .env"); sys.exit(1)

    print("\nAuthenticating with CDSE ...")
    token = get_access_token(username, password)
    print("  Token obtained.\n")

    downloaded = [download_product(p, token) for p in products]
    print(f"\nDone. {len(downloaded)} file(s) in {cfg.DATA_RAW}")

if __name__ == "__main__":
    main()
