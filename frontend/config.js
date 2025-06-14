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

function formatSources(sources) {
    if (!sources || sources.length === 0) return '';
    
    let sourcesHtml = '<div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">';
    sourcesHtml += '<details style="margin-top: 5px;">';
    sourcesHtml += '<summary style="cursor: pointer; color: var(--gold);">📚 Источники из архивов White Star Line</summary>';
    
    sources.forEach((source, index) => {
        sourcesHtml += `<div style="margin: 5px 0; padding: 5px; background: rgba(212, 175, 55, 0.1); border-radius: 5px;">`;
        sourcesHtml += `<strong>${source.topic || 'Архивы'}:</strong> ${source.content}`;
        sourcesHtml += `</div>`;
    });
    
    sourcesHtml += '</details></div>';
    return sourcesHtml;
}
