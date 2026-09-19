import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        timeout: 180_000,
        proxyTimeout: 180_000,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/results": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        timeout: 60_000,
      },
    },
  },
});
