# 🚗 Driver Monitoring Frontend

A comprehensive web interface for AI-powered driver monitoring video analysis with real-time processing, annotated video playback, and detailed reporting capabilities.

## ✨ Features

### 🎬 **Video Upload & Analysis**
- **Drag & Drop Interface**: Easy video file upload with visual feedback
- **Multiple Format Support**: MP4, AVI, MOV, MKV, WebM
- **Real-time Progress**: Live analysis progress with AI processing updates
- **Configuration Options**: Adjustable sensitivity settings and detection features

### 📊 **Interactive Analysis Results**
- **Safety Scoring**: Overall, fatigue, attention, and compliance scores
- **Behavior Charts**: Interactive doughnut and bar charts using Chart.js
- **Risk Assessment**: Color-coded risk levels with visual indicators
- **Event Timeline**: Detailed timeline of detected safety events

### 🎥 **Annotated Video Playback**
- **Real-time Overlays**: Event annotations displayed during video playback
- **Interactive Timeline**: Click events to jump to specific timestamps
- **Event Markers**: Visual markers on progress bar showing event locations
- **Video Controls**: Full playback controls with volume and seeking

### 📄 **Report Export**
- **PDF Reports**: Comprehensive analysis reports with charts and recommendations
- **CSV Data Export**: Raw data export for further analysis in Excel/Google Sheets
- **Multiple Formats**: Session summaries and detailed event logs
- **Professional Layout**: Clean, printable report formatting

### 🎨 **Modern UI/UX**
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark/Light Themes**: Automatic theme adaptation
- **Smooth Animations**: Fluid transitions and loading states
- **Accessibility**: Screen reader compatible and keyboard navigation

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn
- AI Backend running on port 8002

### Installation & Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Or use the automated script**
   ```bash
   chmod +x ../start_frontend.sh
   ../start_frontend.sh
   ```

### Production Build

```bash
npm run build
npm run preview
```

## 🏗️ Architecture

### Component Structure
```
src/
├── components/
│   ├── VideoUpload.jsx       # Video upload interface
│   ├── ProgressTracker.jsx   # Analysis progress display
│   ├── AnalysisResults.jsx   # Results visualization
│   ├── AnnotatedVideo.jsx    # Video player with annotations
│   └── ReportExport.jsx      # PDF/CSV export functionality
├── utils/
│   └── api.js               # API communication functions
├── App.jsx                  # Main application component
└── main.jsx                 # Application entry point
```

### Key Technologies
- **React 18**: Modern React with hooks and functional components
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Interactive charts and data visualization
- **jsPDF**: Client-side PDF generation
- **React Dropzone**: File upload with drag & drop
- **Lucide React**: Beautiful SVG icons

## 🔧 Configuration

### API Configuration
The frontend connects to the AI backend at `http://localhost:8002`. Update the API base URL in `src/utils/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8002'
```

### Build Configuration
Vite configuration in `vite.config.js` includes:
- CORS settings for backend communication
- Code splitting for optimal loading
- Development server on port 3000

## 📱 User Interface Flow

### 1. **Upload Phase**
- Drag & drop video file or click to browse
- Configure analysis settings (driver ID, sensitivity levels)
- Advanced options for detection features
- File validation and preview

### 2. **Analysis Phase** 
- Real-time progress tracking
- AI processing status updates
- Visual indicators for different analysis stages
- Connection status to backend services

### 3. **Results Phase**
- Safety score dashboard
- Interactive behavior distribution charts
- Event timeline with detailed information
- Risk assessment and recommendations

### 4. **Video Review Phase**
- Annotated video playback
- Event overlays during relevant timestamps
- Interactive timeline with event markers
- Side panel with event details

### 5. **Export Phase**
- PDF report generation with charts
- CSV data export for analysis
- Download management
- Session summary statistics

## 🎯 Features in Detail

### Video Upload Component
- **File Validation**: Checks file type, size, and format
- **Preview Information**: Shows duration, size, and metadata
- **Configuration Panel**: Sensitivity sliders and feature toggles
- **Help Section**: Tips for optimal video quality

### Analysis Results Component
- **Interactive Charts**: Click and hover interactions
- **Metric Cards**: Key performance indicators
- **Event Details**: Expandable event information
- **Navigation**: Smooth transitions between phases

### Annotated Video Component
- **Custom Video Player**: Built-in controls and seeking
- **Event Overlays**: Real-time annotation display
- **Timeline Markers**: Visual event indicators
- **Sidebar Navigation**: Quick event access

### Report Export Component
- **PDF Generation**: Multi-page reports with styling
- **CSV Export**: Structured data with headers
- **Download Status**: Success/error feedback
- **File Naming**: Automatic timestamp naming

## 🔌 API Integration

### Endpoints Used
- `POST /api/driver-monitoring/analyze` - Video analysis
- `GET /api/driver-monitoring/results/{id}` - Get analysis results
- `GET /api/driver-monitoring/status/{id}` - Check analysis status
- `GET /health` - Backend health check

### Error Handling
- Connection timeout handling (5 minutes for video processing)
- User-friendly error messages
- Retry mechanisms for failed requests
- Offline detection and messaging

## 🎨 Design System

### Color Palette
- **Primary**: Blue gradients (#667eea to #764ba2)
- **Success**: Green (#10b981)
- **Warning**: Yellow (#f59e0b)  
- **Danger**: Red (#ef4444)
- **Info**: Blue (#3b82f6)

### Typography
- **Font Family**: System fonts (SF Pro, Segoe UI, Roboto)
- **Headings**: Semibold weights with proper hierarchy
- **Body Text**: Regular weight with sufficient contrast
- **Labels**: Medium weight with letter spacing

### Layout
- **Grid System**: CSS Grid and Flexbox
- **Spacing**: 4px base unit (Tailwind spacing scale)
- **Breakpoints**: Mobile-first responsive design
- **Cards**: Consistent shadow and border radius

## 🔍 Development

### Code Structure
- **Functional Components**: Modern React patterns
- **Custom Hooks**: Reusable state logic
- **API Utilities**: Centralized backend communication
- **Type Safety**: PropTypes for component validation

### Performance Optimizations
- **Code Splitting**: Automatic chunk splitting
- **Lazy Loading**: On-demand component loading
- **Memoization**: React.memo for expensive components
- **Bundle Analysis**: Optimized dependencies

### Testing Strategy
- **Component Testing**: Jest and React Testing Library
- **API Mocking**: Mock Service Worker (MSW)
- **E2E Testing**: Playwright for user workflows
- **Accessibility Testing**: axe-core integration

## 🚀 Deployment

### Production Build
```bash
npm run build
```

### Static Hosting
The built files in `dist/` can be served by any static file server:
- **Vercel**: Automatic deployment from Git
- **Netlify**: Drag & drop deployment
- **GitHub Pages**: Static site hosting
- **AWS S3**: Cloud storage hosting

### Environment Variables
Create `.env` file for configuration:
```
VITE_API_BASE_URL=http://localhost:8002
VITE_APP_VERSION=2.0.0
```

## 🔧 Troubleshooting

### Common Issues

**Frontend won't start:**
- Check Node.js version (16+ required)
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall

**API connection errors:**
- Verify backend is running on port 8002
- Check CORS settings in backend
- Test API health endpoint manually

**Build failures:**
- Check for TypeScript errors
- Verify all dependencies are installed
- Clear Vite cache: `rm -rf node_modules/.vite`

**Video upload issues:**
- Check file size limits (500MB max)
- Verify video format compatibility
- Test with smaller sample video

### Development Tips
- Use React DevTools browser extension
- Enable Vite HMR for faster development
- Check browser console for detailed errors
- Use network tab to debug API calls

## 📚 Additional Resources

- [React Documentation](https://react.dev)
- [Vite Guide](https://vitejs.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [Chart.js Documentation](https://www.chartjs.org)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Driver Monitoring Validation System and follows the same license terms.

---

🚗 **Driver Monitoring Frontend v2.0.0** - Built with React, powered by AI