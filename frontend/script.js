let timer;
let minutes = 25;
let seconds = 0;
let isRunning = false;

// DOM elements
const minutesDisplay = document.getElementById("minutes");
const secondsDisplay = document.getElementById("seconds");
const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const taskInput = document.getElementById("taskInput");
const addTaskBtn = document.getElementById("addTaskBtn");
const taskList = document.getElementById("taskList");
const feedbackBtn = document.getElementById("getFeedbackBtn");
const aiFeedback = document.getElementById("aiFeedback");

// Timer functions
function updateDisplay() {
  minutesDisplay.textContent = String(minutes).padStart(2, "0");
  secondsDisplay.textContent = String(seconds).padStart(2, "0");
}

function startTimer() {
  if (!isRunning) {
    isRunning = true;
    timer = setInterval(() => {
      if (seconds === 0) {
        if (minutes === 0) {
          clearInterval(timer);
          alert("Time's up! Take a short break!");
          isRunning = false;
          return;
        }
        minutes--;
        seconds = 59;
      } else {
        seconds--;
      }
      updateDisplay();
    }, 1000);
  }
}

function pauseTimer() {
  clearInterval(timer);
  isRunning = false;
}

function resetTimer() {
  clearInterval(timer);
  minutes = 25;
  seconds = 0;
  isRunning = false;
  updateDisplay();
}

startBtn.addEventListener("click", startTimer);
pauseBtn.addEventListener("click", pauseTimer);
resetBtn.addEventListener("click", resetTimer);

// Task management
addTaskBtn.addEventListener("click", async () => {
  const taskText = taskInput.value.trim();
  if (taskText === "") return;

  const li = document.createElement("li");
  li.textContent = taskText;

  const deleteBtn = document.createElement("button");
  deleteBtn.textContent = "X";
  deleteBtn.onclick = () => li.remove();

  li.appendChild(deleteBtn);
  taskList.appendChild(li);

  taskInput.value = "";

  // Send to backend
  await fetch("http://127.0.0.1:8000/tasks/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: taskText }),
  });
});

// AI Feedback
feedbackBtn.addEventListener("click", async () => {
  aiFeedback.textContent = "Getting feedback...";
  try {
    const res = await fetch("http://127.0.0.1:8000/focus-feedback/");
    const data = await res.json();
    aiFeedback.textContent = data.feedback;
  } catch (error) {
    aiFeedback.textContent = "Error connecting to AI backend.";
  }
});

// Initialize display
updateDisplay();
