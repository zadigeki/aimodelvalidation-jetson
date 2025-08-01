import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ValidationFile, ValidationProgress, ValidationReport, AppSettings, User } from '@/types'

interface AppState {
  // Files
  files: ValidationFile[]
  selectedFile: ValidationFile | null
  
  // Progress tracking
  validationProgress: { [fileId: string]: ValidationProgress }
  
  // Reports
  reports: ValidationReport[]
  
  // Settings
  settings: AppSettings
  isDarkMode: boolean
  
  // User
  user: User | null
  
  // UI State
  isLoading: boolean
  error: string | null
  sidebarOpen: boolean
  
  // Actions
  setFiles: (files: ValidationFile[]) => void
  addFile: (file: ValidationFile) => void
  updateFile: (fileId: string, updates: Partial<ValidationFile>) => void
  removeFile: (fileId: string) => void
  selectFile: (file: ValidationFile | null) => void
  
  setValidationProgress: (fileId: string, progress: ValidationProgress) => void
  clearValidationProgress: (fileId: string) => void
  
  addReport: (report: ValidationReport) => void
  removeReport: (reportId: string) => void
  
  updateSettings: (settings: Partial<AppSettings>) => void
  toggleDarkMode: () => void
  
  setUser: (user: User | null) => void
  
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setSidebarOpen: (open: boolean) => void
  
  // Reset actions
  resetFiles: () => void
  resetProgress: () => void
  reset: () => void
}

const defaultSettings: AppSettings = {
  autoPlay: false,
  showConfidenceScores: true,
  confidenceThreshold: 0.5,
  darkMode: false,
  notifications: true,
  autoExport: false,
  exportFormat: 'json',
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      files: [],
      selectedFile: null,
      validationProgress: {},
      reports: [],
      settings: defaultSettings,
      isDarkMode: false,
      user: null,
      isLoading: false,
      error: null,
      sidebarOpen: true,

      // File actions
      setFiles: (files) => set({ files }),
      
      addFile: (file) => set((state) => ({
        files: [...state.files, file]
      })),
      
      updateFile: (fileId, updates) => set((state) => ({
        files: state.files.map(file => 
          file.id === fileId ? { ...file, ...updates } : file
        ),
        selectedFile: state.selectedFile?.id === fileId 
          ? { ...state.selectedFile, ...updates }
          : state.selectedFile
      })),
      
      removeFile: (fileId) => set((state) => ({
        files: state.files.filter(file => file.id !== fileId),
        selectedFile: state.selectedFile?.id === fileId ? null : state.selectedFile,
        validationProgress: Object.fromEntries(
          Object.entries(state.validationProgress).filter(([id]) => id !== fileId)
        )
      })),
      
      selectFile: (file) => set({ selectedFile: file }),

      // Progress actions
      setValidationProgress: (fileId, progress) => set((state) => ({
        validationProgress: {
          ...state.validationProgress,
          [fileId]: progress
        }
      })),
      
      clearValidationProgress: (fileId) => set((state) => ({
        validationProgress: Object.fromEntries(
          Object.entries(state.validationProgress).filter(([id]) => id !== fileId)
        )
      })),

      // Report actions
      addReport: (report) => set((state) => ({
        reports: [...state.reports, report]
      })),
      
      removeReport: (reportId) => set((state) => ({
        reports: state.reports.filter(report => report.id !== reportId)
      })),

      // Settings actions
      updateSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings },
        isDarkMode: newSettings.darkMode !== undefined ? newSettings.darkMode : state.isDarkMode
      })),
      
      toggleDarkMode: () => set((state) => {
        const newDarkMode = !state.isDarkMode
        return {
          isDarkMode: newDarkMode,
          settings: { ...state.settings, darkMode: newDarkMode }
        }
      }),

      // User actions
      setUser: (user) => set({ user }),

      // UI actions
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),

      // Reset actions
      resetFiles: () => set({ 
        files: [], 
        selectedFile: null 
      }),
      
      resetProgress: () => set({ 
        validationProgress: {} 
      }),
      
      reset: () => set({ 
        files: [],
        selectedFile: null,
        validationProgress: {},
        reports: [],
        isLoading: false,
        error: null
      }),
    }),
    {
      name: 'supervision-ui-storage',
      partialize: (state) => ({
        settings: state.settings,
        isDarkMode: state.isDarkMode,
        sidebarOpen: state.sidebarOpen,
        user: state.user,
      }),
    }
  )
)