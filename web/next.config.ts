import type { NextConfig } from "next";

const API = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // the browser talks to /api/*; Next proxies to FastAPI (no CORS, works from a phone on the LAN)
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  experimental: { proxyClientMaxBodySize: "300mb" },
};

export default nextConfig;
