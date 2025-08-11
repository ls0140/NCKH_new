// DOM Elements
const analyzeBtn = document.getElementById('analyze-btn');
const newsContent = document.getElementById('news-content');
const newsUrl = document.getElementById('news-url');
const results = document.getElementById('results');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceScore = document.getElementById('confidence-score');
const resultIcon = document.getElementById('result-icon');
const resultTitle = document.getElementById('result-title');
const resultDescription = document.getElementById('result-description');
const sourceScore = document.getElementById('source-score');
const languageScore = document.getElementById('language-score');
const timeScore = document.getElementById('time-score');
const linksScore = document.getElementById('links-score');

// Navigation
const navLinks = document.querySelectorAll('.nav-link');

// Smooth scrolling for navigation
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        
        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: 'smooth'
            });
            
            // Update active nav link
            navLinks.forEach(navLink => navLink.classList.remove('active'));
            link.classList.add('active');
        }
    });
});

// Update active nav link on scroll
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const scrollPos = window.scrollY + 100;
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${sectionId}`) {
                    link.classList.add('active');
                }
            });
        }
    });
});

// Analysis functionality
analyzeBtn.addEventListener('click', async () => {
    const content = newsContent.value.trim();
    const url = newsUrl.value.trim();
    
    if (!content && !url) {
        showNotification('Vui lòng nhập nội dung bài báo hoặc URL để phân tích', 'error');
        return;
    }
    
    // Show loading state
    analyzeBtn.innerHTML = '<div class="loading"></div> Đang phân tích...';
    analyzeBtn.disabled = true;
    
    // Simulate AI analysis
    await simulateAnalysis(content || url);
    
    // Reset button
    analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Phân Tích Ngay';
    analyzeBtn.disabled = false;
});

// Simulate AI analysis
async function simulateAnalysis(input) {
    // Show results section
    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth' });
    
    // Reset scores
    confidenceScore.textContent = '0%';
    confidenceFill.style.width = '0%';
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate fake analysis results
    const analysis = generateFakeAnalysis(input);
    
    // Animate confidence score
    animateConfidenceScore(analysis.confidence);
    
    // Update result details
    updateResultDetails(analysis);
    
    // Update factor scores
    updateFactorScores(analysis.factors);
}

// Generate fake analysis results
function generateFakeAnalysis(input) {
    const confidence = Math.random() * 100;
    let result, icon, title, description;
    
    if (confidence > 80) {
        result = 'true';
        icon = 'fas fa-check-circle';
        title = 'Tin Tức Đáng Tin Cậy';
        description = 'Bài báo này có độ tin cậy cao và có thể được tin tưởng.';
    } else if (confidence > 50) {
        result = 'uncertain';
        icon = 'fas fa-exclamation-triangle';
        title = 'Cần Thận Trọng';
        description = 'Bài báo này có một số dấu hiệu đáng ngờ, cần kiểm tra thêm.';
    } else {
        result = 'false';
        icon = 'fas fa-times-circle';
        title = 'Tin Tức Không Đáng Tin Cậy';
        description = 'Bài báo này có nhiều dấu hiệu của tin giả, không nên tin tưởng.';
    }
    
    return {
        confidence: Math.round(confidence),
        result,
        icon,
        title,
        description,
        factors: {
            source: Math.round(Math.random() * 100),
            language: Math.round(Math.random() * 100),
            time: Math.round(Math.random() * 100),
            links: Math.round(Math.random() * 100)
        }
    };
}

// Animate confidence score
function animateConfidenceScore(targetScore) {
    let currentScore = 0;
    const increment = targetScore / 50;
    
    const animation = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(animation);
        }
        
        confidenceScore.textContent = Math.round(currentScore) + '%';
        confidenceFill.style.width = currentScore + '%';
    }, 20);
}

// Update result details
function updateResultDetails(analysis) {
    resultIcon.className = analysis.icon + ' result-' + analysis.result;
    resultTitle.textContent = analysis.title;
    resultDescription.textContent = analysis.description;
}

// Update factor scores
function updateFactorScores(factors) {
    setTimeout(() => {
        sourceScore.textContent = factors.source + '%';
    }, 500);
    
    setTimeout(() => {
        languageScore.textContent = factors.language + '%';
    }, 1000);
    
    setTimeout(() => {
        timeScore.textContent = factors.time + '%';
    }, 1500);
    
    setTimeout(() => {
        linksScore.textContent = factors.links + '%';
    }, 2000);
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'error' ? '#dc3545' : '#667eea'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 10000;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 300px;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Contact form handling
const contactForm = document.querySelector('.contact-form');
contactForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const name = document.getElementById('contact-name').value;
    const email = document.getElementById('contact-email').value;
    const subject = document.getElementById('contact-subject').value;
    const message = document.getElementById('contact-message').value;
    
    if (name && email && message) {
        showNotification('Tin nhắn đã được gửi thành công! Chúng tôi sẽ phản hồi sớm nhất.', 'success');
        contactForm.reset();
    } else {
        showNotification('Vui lòng điền đầy đủ thông tin bắt buộc (họ tên, email, nội dung).', 'error');
    }
});

// Add some sample content for testing
function addSampleContent() {
    const sampleTexts = [
        "Chính phủ Việt Nam vừa công bố kế hoạch phát triển kinh tế mới với mục tiêu tăng trưởng GDP 7% trong năm 2024. Theo báo cáo, các ngành công nghiệp chế biến và xuất khẩu sẽ được ưu tiên phát triển.",
        "Các nhà khoa học tại Đại học Quốc gia Hà Nội đã phát hiện ra phương pháp mới để điều trị bệnh ung thư với tỷ lệ thành công lên đến 95%. Nghiên cứu này đã được công bố trên tạp chí Nature.",
        "Người ngoài hành tinh đã liên lạc với chính phủ Việt Nam và đề nghị hợp tác trong lĩnh vực công nghệ vũ trụ. Theo nguồn tin đáng tin cậy, họ sẽ đến thăm Trái Đất vào tháng tới."
    ];
    
    const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
    newsContent.value = randomText;
}

// Add sample content button (for demo purposes)
const sampleBtn = document.createElement('button');
sampleBtn.textContent = 'Thử Nghiệm Mẫu';
sampleBtn.className = 'sample-btn';
sampleBtn.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #28a745;
    color: white;
    border: none;
    padding: 0.75rem 1rem;
    border-radius: 25px;
    cursor: pointer;
    font-weight: 600;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    z-index: 1000;
    transition: transform 0.3s ease;
`;

sampleBtn.addEventListener('click', addSampleContent);
sampleBtn.addEventListener('mouseenter', () => {
    sampleBtn.style.transform = 'scale(1.05)';
});
sampleBtn.addEventListener('mouseleave', () => {
    sampleBtn.style.transform = 'scale(1)';
});

document.body.appendChild(sampleBtn);

// Add success notification type
const originalShowNotification = showNotification;
showNotification = function(message, type = 'info') {
    if (type === 'success') {
        const notification = document.createElement('div');
        notification.className = 'notification notification-success';
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-check-circle"></i>
                <span>${message}</span>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 10000;
            transform: translateX(400px);
            transition: transform 0.3s ease;
            max-width: 300px;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(400px)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 5000);
    } else {
        originalShowNotification(message, type);
    }
};

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    console.log('AI Fake News Detection Website loaded successfully!');
    console.log('This is a demo version with simulated AI analysis.');
    console.log('Click "Thử Nghiệm Mẫu" to test with sample content.');
}); 