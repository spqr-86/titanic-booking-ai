// Конфигурация API
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000',
    ENDPOINTS: {
        CHAT: '/api/chat/message',
        HEALTH: '/api/health',
        CLEAR_SESSION: '/api/chat/session'
    }
};

// Утилиты для работы с сессиями
function generateSessionId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substr(2, 9);
    return `titanic_${timestamp}_${random}`;
}

function getSessionId() {
    let sessionId = localStorage.getItem('titanic_session_id');
    if (!sessionId) {
        sessionId = generateSessionId();
        localStorage.setItem('titanic_session_id', sessionId);
    }
    return sessionId;
}

// Проверка здоровья API
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`);
        const data = await response.json();
        console.log('✅ API Status:', data);
        return data.status === 'healthy' && data.openai_status === 'configured';
    } catch (error) {
        console.error('❌ API недоступен:', error);
        return false;
    }
}

// Очистка сессии чата
async function clearChatSession() {
    try {
        const sessionId = getSessionId();
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CLEAR_SESSION}/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Создаем новую сессию
            localStorage.removeItem('titanic_session_id');
            console.log('🗑️ Сессия чата очищена');
            return true;
        }
    } catch (error) {
        console.error('Ошибка при очистке сессии:', error);
    }
    return false;
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', async () => {
    const isHealthy = await checkAPIHealth();
    if (!isHealthy) {
        console.warn('⚠️ API не готов. Убедитесь что backend запущен.');
    }
});
