# AI Model Validation - Supervision UI

A professional React-based frontend interface for AI model validation and video/image supervision.

## Features

- **Drag & Drop Upload**: Intuitive file upload with progress tracking
- **Real-time Validation**: Live progress updates via WebSocket
- **Video Player**: Frame-by-frame analysis with annotation display
- **Results Visualization**: Interactive results with confidence filtering
- **Export Options**: Multiple export formats (JSON, CSV, XML)
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Mode**: Built-in dark/light theme support

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **React Query** for API state management
- **Zustand** for global state
- **React Hook Form** for form handling
- **Socket.IO** for real-time updates

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run type-check` - Run TypeScript checks

## API Integration

The frontend expects a backend API at `http://localhost:8000` with the following endpoints:

- `POST /api/files/upload` - File upload
- `GET /api/files` - List files
- `GET /api/files/:id` - Get file details
- `POST /api/files/:id/validate` - Start validation
- `GET /api/files/:id/results` - Get validation results
- `WebSocket /ws` - Real-time updates

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### Proxy Configuration

The Vite config includes proxy settings for development:

```typescript
server: {
  proxy: {
    '/api': 'http://localhost:8000',
    '/ws': {
      target: 'ws://localhost:8000',
      ws: true,
    },
  },
}
```

## File Structure

```
src/
├── components/           # React components
│   ├── Dashboard/       # Dashboard overview
│   ├── Layout/          # App layout components
│   ├── Results/         # Results visualization
│   ├── Upload/          # File upload interface
│   ├── Validation/      # Validation controls
│   └── Video/           # Video player components
├── services/            # API and WebSocket services
├── store/               # Zustand state management
├── types/               # TypeScript type definitions
├── utils/               # Utility functions
└── styles/              # Global styles
```

## Component Overview

### Core Components

- **Layout**: Main app layout with sidebar and header
- **Dashboard**: System overview and quick actions
- **UploadDropzone**: Drag-and-drop file upload
- **VideoPlayer**: Video playback with frame controls
- **ResultsViewer**: Interactive results visualization
- **ValidationProgress**: Real-time validation progress

### Upload Flow

1. User drags files to UploadDropzone
2. Files are validated and uploaded with progress tracking
3. Upload completion triggers processing status
4. Files become available for validation

### Validation Flow

1. User selects file and starts validation
2. WebSocket connection provides real-time progress
3. Progress updates through multiple stages
4. Completion triggers results display

### Results Display

1. Interactive annotation overlay
2. Confidence filtering and class selection
3. Object details and statistics
4. Export options in multiple formats

## Customization

### Styling

The app uses Tailwind CSS with a custom design system:

- Primary colors: Blue palette
- Success: Green palette  
- Warning: Yellow palette
- Error: Red palette

Modify `tailwind.config.js` to customize the theme.

### API Integration

Update `src/services/api.ts` to modify API endpoints and request handling.

### State Management

The app uses Zustand for state management. Main stores:

- `appStore`: Global application state
- File management and validation progress
- User settings and preferences

## Deployment

### Production Build

```bash
npm run build
```

The build artifacts will be in the `dist/` directory.

### Docker

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment Variables

Set these in production:

- `VITE_API_URL`: Backend API URL
- `VITE_WS_URL`: WebSocket URL

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- Code splitting with React.lazy()
- Image optimization with lazy loading
- Virtual scrolling for large lists
- Efficient re-renders with React.memo()

## Accessibility

- ARIA labels and roles
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.