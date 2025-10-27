# 📚 LeRobot Web GUI Documentation

## 🚀 **Quick Start**

### **📦 Installation**
```bash
# 1. Install LeRobot with web GUI support (Python dependencies)
pip install -e .[web]

# 2. Install Node.js dependencies (required for Vue.js frontend)
cd frontend
npm install
cd ..
```

### **📋 Prerequisites**
- **Python 3.10+** with LeRobot framework
- **Node.js 16+** for Vue.js frontend
- **npm** package manager (comes with Node.js)

### **🎯 One-Command Launch (LeLab Style)**
```bash
# Start GUI (FastAPI + Vue.js + Auto-browser)
lerobot-gui

# Development mode with backend selection
lerobot-gui-dev

# Create desktop shortcut
lerobot-gui-shortcut
```

### **🔧 Advanced Usage**
```bash
# Backend only
lerobot-gui-backend

# Frontend only  
lerobot-gui-frontend

# System status
lerobot-gui-status
```

### **🖥️ Desktop Integration**
```bash
# Create desktop shortcut for easy access
lerobot-gui-shortcut

# Then click the desktop icon to start GUI
```

## 📖 **Development History**
- **[FastTrack Implementation Plan](FastTrack_Implementation_Plan.md)** - Complete development journey and all code changes implemented

## 🔍 **Research & Analysis** 
- **[GUI Comparison Analysis](GUI_Comparison_Analysis.md)** - Comparison with LeLab project
- **[LeLab Project Analysis](LeLab_Project_Analysis.md)** - Research on LeLab architecture

---

## 🎯 **Current Status**

✅ **Phase 1 Complete**: Simplified configuration, enhanced emergency stop, auto-start script  
✅ **Phase 2 Complete**: FastAPI migration with Socket.IO compatibility  
🚀 **Ready for**: Phase 3 (Quick Wins) or advanced features

### **How to Start Development**

#### **🚀 Quick Start (LeLab Style - Recommended)**
```bash
# Install with web dependencies
pip install -e .[web]

# Start GUI (one command, auto-browser)
lerobot-gui
```

#### **🔧 Advanced Development**
```bash
# Development mode with backend selection
lerobot-gui-dev                              # FastAPI (default)
# (Legacy Flask backend removed)
lerobot-gui-dev --backend fastapi            # FastAPI (explicit)
```

#### **📜 Script-Based (Alternative)**
```bash
cd web\scripts
python start_dev.py                         # FastAPI only
python start_dev_advanced.py                # Backend selection
```

This launches:
- FastAPI backend on `http://localhost:5000`
- Vue.js frontend on `http://localhost:5173` 
- Interactive API docs at `http://localhost:5000/api/docs`

### **Key Features Implemented**
- **Simplified Configuration**: Preset dropdown (Safe/Normal/Performance)
- **Enhanced Emergency Stop**: Space key + button with API fallback
- **FastAPI Backend**: Async performance with auto-documentation
- **Socket.IO Integration**: Real-time communication
- **CLI Commands**: LeLab-style one-command launch
- **Desktop Integration**: Cross-platform shortcut creation
- **Auto-Start Scripts**: One-command development environment
- **Mock Mode**: Hardware-free development and testing

### **🎯 LeLab Integration Benefits**
- **One-Command Launch**: `lerobot-gui` starts everything
- **Auto-Dependencies**: Automatically installs web requirements
- **Desktop Shortcuts**: Cross-platform GUI shortcuts
- **CLI Interface**: Professional command-line tools
- **Integrated Architecture**: No separate repository management

### **Next Steps**
- Phase 3 Quick Wins: Better error handling UI, development tools panel
- Advanced Features: Background tasks, authentication, database integration
- Production Ready: Security features, deployment configuration
