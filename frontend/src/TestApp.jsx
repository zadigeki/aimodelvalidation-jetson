import React from 'react'

function TestApp() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1 style={{ color: '#333' }}>🚗 Driver Monitoring System - Test</h1>
      <p style={{ color: '#666' }}>If you can see this, React is working correctly!</p>
      <div style={{ 
        padding: '20px', 
        background: '#f0f0f0', 
        borderRadius: '8px', 
        marginTop: '20px' 
      }}>
        <h2>System Status</h2>
        <ul>
          <li>✅ React is rendering</li>
          <li>✅ JavaScript is working</li>
          <li>✅ Frontend server is operational</li>
        </ul>
      </div>
    </div>
  )
}

export default TestApp