import { useEffect, useRef, useState } from "react";
import { Expand, MapPinned } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PipelineResult } from "@/lib/types";

export function ScientificMap({ result }: { result?: PipelineResult | undefined }) {
  const mapNode = useRef<HTMLDivElement>(null);
  const mapRef = useRef<{ remove: () => void } | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function createMap() {
      const maplibregl = await import("maplibre-gl");
      if (cancelled || !mapNode.current) return;
      const map = new maplibregl.Map({
        container: mapNode.current,
        style: {
          version: 8,
          sources: {
            base: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, attribution: "© OpenStreetMap" },
          },
          layers: [{ id: "base", type: "raster", source: "base", paint: { "raster-saturation": -1, "raster-brightness-max": 0.42, "raster-contrast": 0.25 } }],
        },
        center: [78.9, 22.5],
        zoom: 3.15,
        attributionControl: false,
      });
      map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "bottom-right");
      map.on("load", () => setLoaded(true));
      mapRef.current = map;
    }
    void createMap();
    return () => { cancelled = true; mapRef.current?.remove(); mapRef.current = null; };
  }, []);

  const hasCoordinates = result?.latitude !== undefined && result.longitude !== undefined;
  return <div className="relative min-h-72 overflow-hidden border border-border bg-map"><div ref={mapNode} className="absolute inset-0" aria-label="Scientific basemap" /><div className="map-grid" aria-hidden /><div className="absolute left-4 top-4 z-10 border border-border bg-background/85 px-3 py-2 backdrop-blur"><div className="data-label">Geospatial operations view</div><div className="mt-1 font-mono text-xs">{loaded ? "BASEMAP READY" : "LOADING BASEMAP"}</div></div><Button className="absolute right-3 top-3 z-10" variant="ghost" size="icon" aria-label="Expand map"><Expand /></Button>{!hasCoordinates && <div className="absolute inset-x-4 bottom-4 z-10 flex items-start gap-3 border border-status-warn/30 bg-background/90 p-3 backdrop-blur"><MapPinned className="mt-0.5 size-4 shrink-0 text-status-warn"/><div><div className="text-[10px] font-bold uppercase">Geographic placement withheld</div><p className="mt-1 text-[10px] text-muted-foreground">The backend provides mesh row/column only. No latitude or longitude has been inferred.</p></div></div>}</div>;
}