import type { NextConfig } from "next";

const backendUrl =
  process.env.BACKEND_URL ||
  `http://localhost:${process.env.BACKEND_PORT || "8000"}`;

const nextConfig: NextConfig = {
  // Use Turbopack (Next.js 16 default)
  // Empty config silences warning about webpack config
  turbopack: {},

  // Default Node proxy timeout is 30s. Analysis endpoints for huge datasets
  // (e.g. droid_1.0.1 with ~95k episodes) routinely run 1–3 minutes while
  // pulling parquet files from HuggingFace; without this they ECONNRESET
  // and surface as HTTP 500 in the frontend even though the backend is fine.
  experimental: {
    proxyTimeout: 10 * 60 * 1000,
  },

  // Same-origin proxy: browser hits /api/*, Next forwards to backend.
  // Removes cross-origin requests entirely, so CORS allowlists and the
  // backend port are no longer the frontend's concern.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }];
  },

  // Keep webpack config for fallback/non-Turbopack builds
  webpack: (config, { isServer }) => {
    // Enable WASM
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    // Handle WASM files
    config.module.rules.push({
      test: /\.wasm$/,
      type: "webassembly/async",
    });

    // Fix for client-side only packages
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }

    return config;
  },
};

export default nextConfig;
