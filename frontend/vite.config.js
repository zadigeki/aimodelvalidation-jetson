import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    cors: {
      origin: ['http://localhost:8002', 'http://localhost:8001'],
      credentials: true,
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['chart.js', 'react-chartjs-2'],
          utils: ['axios', 'jspdf', 'html2canvas'],
        }
      }
    }
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'axios', 'chart.js', 'react-chartjs-2']
  }
})