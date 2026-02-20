// Starfield Animation
const canvas = document.getElementById('canvas-stars');
const ctx = canvas.getContext('2d');

let w, h, stars = [];

function init() {
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w;
    canvas.height = h;

    stars = [];
    for (let i = 0; i < 200; i++) {
        stars.push({
            x: Math.random() * w,
            y: Math.random() * h,
            size: Math.random() * 2,
            speed: Math.random() * 0.5 + 0.1
        });
    }
}

function animate() {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff';

    stars.forEach(s => {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fill();
        s.y -= s.speed;
        if (s.y < 0) s.y = h;
    });

    requestAnimationFrame(animate);
}

window.addEventListener('resize', init);
init();
animate();

// Section Navigation
function showSection(sectionId) {
    // Hide all sections
    document.getElementById('section-overview').style.display = 'none';
    document.getElementById('section-architecture').style.display = 'none';
    document.getElementById('section-deployment').style.display = 'none';

    // Show target section
    document.getElementById('section-' + sectionId).style.display = 'block';

    // Update nav-item active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('onclick').includes(sectionId)) {
            item.classList.add('active');
        }
    });
}

// Lucide Icons
lucide.createIcons();

// Chart.js Configuration
const chartCtx = document.getElementById('evalChart').getContext('2d');
new Chart(chartCtx, {
    type: 'line',
    data: {
        labels: ['Jan 20', 'Jan 25', 'Feb 1', 'Feb 8', 'Feb 15', 'Feb 20'],
        datasets: [{
            label: 'Faithfulness Score',
            data: [0.82, 0.85, 0.89, 0.92, 0.93, 0.94],
            borderColor: '#00f2ff',
            backgroundColor: 'rgba(0, 242, 255, 0.1)',
            fill: true,
            tension: 0.4
        }, {
            label: 'Relevancy Score',
            data: [0.75, 0.78, 0.81, 0.88, 0.90, 0.92],
            borderColor: '#7000ff',
            backgroundColor: 'rgba(112, 0, 255, 0.1)',
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#94a3b8' }
            }
        },
        scales: {
            y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#94a3b8' }
            }
        }
    }
});
