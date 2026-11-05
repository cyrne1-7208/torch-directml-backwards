# Copyright (c) 2026 Cyrne1. MIT License; see LICENSE.
"""Quote live prices for a Spot n1-standard-2, T4, Windows and 50 GiB disk."""

import argparse
import json
import os
import subprocess
import urllib.parse
import urllib.request


def gcloud(*args):
    return subprocess.check_output(["gcloud", *args], text=True).strip()


def skus(token):
    page = ""
    while True:
        query = urllib.parse.urlencode(
            {"pageSize": 5000, "pageToken": page, "currencyCode": "USD"}
        )
        request = urllib.request.Request(
            f"https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?{query}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        yield from data.get("skus", [])
        page = data.get("nextPageToken")
        if not page:
            return


def unit_price(sku):
    expression = sku["pricingInfo"][-1]["pricingExpression"]
    # Use the highest tier, so a free allowance cannot understate the CI budget.
    rates = [
        int(r["unitPrice"].get("units", 0)) + r["unitPrice"].get("nanos", 0) / 1e9
        for r in expression["tieredRates"]
    ]
    return max(rates), expression


def ranked_zones(catalog, zones):
    parts = {}
    windows = []
    ip = []
    for sku in catalog:
        description = sku["description"]
        spot = sku["category"]["usageType"] == "Preemptible"
        if spot and description.startswith("Nvidia Tesla T4 GPU attached to Spot"):
            price, expression = unit_price(sku)
            part, hourly = "gpu", price
        elif spot and description.startswith(
            "Spot Preemptible N1 Predefined Instance Core"
        ):
            price, expression = unit_price(sku)
            part, hourly = "cpu", price * 2
        elif spot and description.startswith(
            "Spot Preemptible N1 Predefined Instance Ram"
        ):
            price, expression = unit_price(sku)
            part, hourly = "ram", price * 7.5
        elif description.startswith("Balanced PD Capacity in"):
            price, expression = unit_price(sku)
            part = "disk"
            hourly = (
                price * 50 * 1024**3 * 3600 / expression["baseUnitConversionFactor"]
            )
        else:
            if (
                "Licensing Fee for Windows Server" in description
                and "CPU cost" in description
                and "BYOL" not in description
            ):
                windows.append(unit_price(sku)[0] * 2)
            if description == "External IP Charge on a Spot Preemptible VM":
                ip.append(unit_price(sku)[0])
            continue
        expected_unit = {"gpu": "h", "cpu": "h", "ram": "GiBy.h", "disk": "GiBy.mo"}[
            part
        ]
        if expression["usageUnit"] != expected_unit:
            raise ValueError(f"Unexpected billing unit for {description}")
        for region in sku.get("serviceRegions", []):
            values = parts.setdefault(region, {})
            values[part] = max(values.get(part, 0), hourly)
    if not windows or not ip:
        raise ValueError("Cannot quote Windows licensing and external IPv4")
    # Windows Server licenses are per vCPU; reserve the highest published rate.
    overhead = max(windows) + max(ip)
    result = []
    for zone in zones:
        region = zone.rsplit("-", 1)[0]
        values = parts.get(region, {})
        if len(values) == 4:
            result.append((sum(values.values()) + overhead, zone))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hourly-usd", type=float, default=0.30)
    parser.add_argument(
        "--zones",
        default=os.environ.get("GCP_ZONES", ""),
        help="Optional comma-separated zone allowlist",
    )
    args = parser.parse_args()
    available = gcloud(
        "compute",
        "accelerator-types",
        "list",
        "--filter=name~^nvidia-tesla-t4$",
        "--format=value(zone.basename())",
    ).split()
    allowed = set(args.zones.split(",")) if args.zones else set(available)
    token = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN") or gcloud(
        "auth", "print-access-token"
    )
    ranked = ranked_zones(skus(token), set(available) & allowed)
    eligible = [
        (price, zone) for price, zone in ranked if 0 < price <= args.max_hourly_usd
    ]
    if not eligible:
        parser.exit(2, "error: no priced T4 zone is within the hourly limit\n")
    for price, zone in eligible:
        print(f"{zone}\t{price:.6f}")


if __name__ == "__main__":
    main()
