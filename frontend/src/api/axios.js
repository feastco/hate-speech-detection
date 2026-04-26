// D:\www\hate-speech\frontend\src\api\axios.js
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Intercept error untuk logging
api.interceptors.response.use(
  response => response,
  error => {
    console.error('[API Error]', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export const predictText = async (text, modelType = 'svm') => {
  const response = await api.post('/predict', { text, model_type: modelType })
  return response.data
}

export const getModels = async () => {
  const response = await api.get('/models')
  return response.data
}

export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}

export default api
