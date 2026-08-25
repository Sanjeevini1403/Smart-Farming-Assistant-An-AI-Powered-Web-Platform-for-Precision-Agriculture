// COMPLETE voice.js file
const commands = {
    "அரிசி விலை": "/market_price?crop=rice&state=Tamil Nadu",
    "மண் பரிந்துரை": "/soil/crop-recommend", 
    "நோய்": "/plant/",
    "வேதியல்": "/chemical/chemical_scan",
    "அரிசி": "/market_price?crop=rice",
    "rice price": "/market_price?crop=rice"
};

let recognition;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'ta-IN';
    
    recognition.onresult = function(event) {
        const command = event.results[0][0].transcript.toLowerCase();
        console.log("🎤 நீ சொன்னது:", command);
        
        for (let key in commands) {
            if (command.includes(key.toLowerCase())) {
                window.location.href = commands[key];
                return;
            }
        }
        alert("புரியல - மீண்டும் சொல்லுங்க! 🎤");
    };
    
    recognition.onend = function() {
        document.getElementById('voiceBtn').classList.remove('listening');
    };
}

document.getElementById('voiceBtn')?.addEventListener('click', function() {
    if (recognition) {
        this.classList.add('listening');
        recognition.start();
    } else {
        alert('Voice not supported in this browser! Use Chrome.');
    }
});
