import { demoData } from './integrate_demo_data.js';

class InverseScalingDemo {
    constructor() {
        this.currentModel = null;
        this.currentTask = null;
        this.currentBudget = 0;
        this.currentInstance = 0;
        
        this.init();
    }

    init() {
        this.populateModelSelect();
        this.populateTaskSelect();
        this.setupEventListeners();
    }

    populateModelSelect() {
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        
        // Group models by category
        const mainModels = demoData.models.filter(m => m.category === 'main');
        const claudeModels = demoData.models.filter(m => m.category === 'claude');
        const openaiModels = demoData.models.filter(m => m.category === 'openai');
        const opensourceModels = demoData.models.filter(m => m.category === 'opensource');
        
        // Add main models first (paper primary models)
        if (mainModels.length > 0) {
            const mainGroup = document.createElement('optgroup');
            mainGroup.label = '📄 Claude Models';
            mainModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                mainGroup.appendChild(option);
            });
            modelSelect.appendChild(mainGroup);
        }
        
        // Add other model categories
        const categories = [
            { models: claudeModels, label: '🔸 Claude Models' },
            { models: openaiModels, label: '🔹 OpenAI Models' },
            { models: opensourceModels, label: '🔓 Open Source Models' }
        ];
        
        categories.forEach(category => {
            if (category.models.length > 0) {
                const group = document.createElement('optgroup');
                group.label = category.label;
                category.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    group.appendChild(option);
                });
                modelSelect.appendChild(group);
            }
        });
    }

    populateTaskSelect() {
        const taskSelect = document.getElementById('taskSelect');
        taskSelect.innerHTML = '<option value="">Select a task...</option>';
        
        // Group tasks by failure mode categories
        const redHerringTasks = demoData.tasks.filter(t => t.category === 'red_herring');
        const correlationTasks = demoData.tasks.filter(t => t.category === 'correlation_overfitting');
        const constraintTasks = demoData.tasks.filter(t => t.category === 'constraint_tracking');
        const aiRiskTasks = demoData.tasks.filter(t => t.category === 'ai_risk');
        
        // Add task categories based on failure modes
        const categories = [
            { tasks: redHerringTasks, label: '🎯 Simple counting tasks with distractors', description: 'Simple counting tasks with distractors' },
            { tasks: correlationTasks, label: '📈 Regression tasks with spurious features', description: 'Regression tasks with spurious features' },
            { tasks: constraintTasks, label: '🧩 Deduction tasks with constraint tracking', description: 'Deduction tasks with constraint tracking' },
            { tasks: aiRiskTasks, label: '⚠️ AI Safety Tasks', description: 'Model-written evaluation' }
        ];
        
        categories.forEach(category => {
            if (category.tasks.length > 0) {
                const group = document.createElement('optgroup');
                group.label = category.label;
                category.tasks.forEach(task => {
                    const option = document.createElement('option');
                    option.value = task.id;
                    option.textContent = task.name;
                    group.appendChild(option);
                });
                taskSelect.appendChild(group);
            }
        });
    }

    setupEventListeners() {
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            this.currentModel = e.target.value;
            this.updateDisplay();
        });

        document.getElementById('taskSelect').addEventListener('change', (e) => {
            this.currentTask = e.target.value;
            this.updateDisplay();
        });
    }

    updateDisplay() {
        if (this.currentModel && this.currentTask) {
            this.showBudgetTabs();
            this.showPrediction();
        } else {
            this.hideBudgetTabs();
            this.hidePrediction();
        }
    }

    showBudgetTabs() {
        const budgetTabs = document.getElementById('budgetTabs');
        budgetTabs.style.display = 'flex';
        budgetTabs.innerHTML = '';

        // Get available budgets for current model and task
        const availableBudgets = this.getAvailableBudgets();
        const maxBudget = Math.max(...availableBudgets);

        availableBudgets.forEach((budget, index) => {
            const tab = document.createElement('button');
            tab.className = `budget-tab ${budget === this.currentBudget ? 'active' : ''}`;
            
            // Highlight comparison between lowest and highest
            if (budget === 0) {
                tab.textContent = 'No Thinking (Baseline)';
                tab.style.backgroundColor = budget === this.currentBudget ? 'var(--primary-orange)' : '#E5E7EB';
            } else if (budget === maxBudget) {
                tab.textContent = `${budget} tokens (Max)`;
                tab.style.backgroundColor = budget === this.currentBudget ? 'var(--primary-orange)' : '#FEF3C7';
                tab.style.border = '2px solid var(--accent-orange)';
            } else {
                tab.textContent = `${budget} tokens`;
            }
            
            tab.addEventListener('click', () => {
                this.currentBudget = budget;
                this.updateBudgetTabs();
                this.showPrediction();
            });
            budgetTabs.appendChild(tab);
        });

        // Auto-select first available budget if current budget is not available
        if (!availableBudgets.includes(this.currentBudget) && availableBudgets.length > 0) {
            this.currentBudget = availableBudgets[0];
        }
    }

    getAvailableBudgets() {
        if (!this.currentModel || !this.currentTask) {
            return demoData.reasoningBudgets;
        }

        const taskData = demoData.predictions[this.currentTask];
        if (!taskData || !taskData[this.currentModel]) {
            return demoData.reasoningBudgets;
        }

        // Get budgets that have data for current model/task combination
        const availableBudgets = [];
        for (const budget of demoData.reasoningBudgets) {
            const budgetData = taskData[this.currentModel][budget.toString()];
            if (budgetData && budgetData.length > 0) {
                availableBudgets.push(budget);
            }
        }

        return availableBudgets.length > 0 ? availableBudgets : [0]; // Always show at least budget 0
    }

    updateBudgetTabs() {
        const tabs = document.querySelectorAll('.budget-tab');
        const availableBudgets = this.getAvailableBudgets();
        const maxBudget = Math.max(...availableBudgets);
        
        tabs.forEach((tab, index) => {
            const budget = availableBudgets[index];
            const isActive = budget === this.currentBudget;
            
            tab.classList.toggle('active', isActive);
            
            // Reset and reapply styling
            if (budget === 0) {
                tab.style.backgroundColor = isActive ? 'var(--primary-orange)' : '#E5E7EB';
                tab.style.color = isActive ? 'var(--white)' : 'var(--text-secondary)';
                tab.style.border = '1px solid var(--border-gray)';
            } else if (budget === maxBudget) {
                tab.style.backgroundColor = isActive ? 'var(--primary-orange)' : '#FEF3C7';
                tab.style.color = isActive ? 'var(--white)' : 'var(--text-primary)';
                tab.style.border = '2px solid var(--accent-orange)';
            } else {
                tab.style.backgroundColor = isActive ? 'var(--primary-orange)' : 'var(--white)';
                tab.style.color = isActive ? 'var(--white)' : 'var(--text-secondary)';
                tab.style.border = '1px solid var(--border-gray)';
            }
        });
    }

    hideBudgetTabs() {
        document.getElementById('budgetTabs').style.display = 'none';
    }

    showPrediction() {
        const predictionDisplay = document.getElementById('predictionDisplay');
        
        // Get prediction data for current budget
        const taskData = demoData.predictions[this.currentTask];
        if (!taskData || !taskData[this.currentModel] || !taskData[this.currentModel][this.currentBudget]) {
            // Check if this is Claude Opus 4 on AI risk tasks - show special message
            const model = demoData.models.find(m => m.id === this.currentModel);
            const task = demoData.tasks.find(t => t.id === this.currentTask);
            
            
            // Show a message for missing data
            predictionDisplay.innerHTML = `
                <div class="no-data-container" style="background: #F3F4F6; border: 1px solid #D1D5DB; border-radius: 8px; padding: 2rem; margin: 1rem 0; text-align: center;">
                    <div style="font-size: 1.25rem; color: #6B7280; margin-bottom: 0.5rem;">
                        No data available for this combination
                    </div>
                    <p style="margin: 0; color: #9CA3AF; font-size: 0.875rem;">
                        This model/task combination doesn't have prediction data in the demo.<br>
                        Currently, only <strong>Synthetic Misleading Math</strong> and <strong>Synthetic Misleading Python</strong> tasks have demo data available.
                    </p>
                </div>
            `;
            predictionDisplay.style.display = 'block';
            return;
        }

        const currentPredictions = taskData[this.currentModel][this.currentBudget];
        if (currentPredictions.length === 0) {
            this.hidePrediction();
            return;
        }

        // Also get prediction data for baseline (0 budget) for comparison
        const baselinePredictions = taskData[this.currentModel][0];
        
        const currentPrediction = currentPredictions[this.currentInstance % currentPredictions.length];
        const baselinePrediction = baselinePredictions && baselinePredictions.length > 0 ? 
            baselinePredictions[this.currentInstance % baselinePredictions.length] : null;
        
        const model = demoData.models.find(m => m.id === this.currentModel);
        const task = demoData.tasks.find(t => t.id === this.currentTask);

        predictionDisplay.innerHTML = this.renderComparisonPredictions(currentPrediction, baselinePrediction, model, task);
        predictionDisplay.style.display = 'block';
    }

    renderComparisonPredictions(currentPrediction, baselinePrediction, model, task) {
        const taskData = demoData.predictions[this.currentTask];
        const currentPredictions = taskData[this.currentModel][this.currentBudget];
        const isOSeriesModel = model.id.includes('o3') || model.id.includes('o4');
        
        // If current budget is 0, show single prediction
        if (this.currentBudget === 0) {
            return this.renderSinglePrediction(currentPrediction, model, task, currentPredictions);
        }
        
        // Show comparison between baseline (0) and current budget
        return `
            ${currentPredictions && currentPredictions.length > 1 ? `
                <div style="display: flex; justify-content: center; align-items: center; margin-top: 1rem; margin-bottom: 1rem; padding-top: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-gray); font-size: 0.875rem; color: var(--text-secondary);">
                    <button onclick="demo.previousInstance()" style="margin-right: 0.5rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border-gray); background: var(--white); border-radius: 0.25rem; cursor: pointer;">←</button>
                    Instance ${this.currentInstance + 1} of ${currentPredictions.length}
                    <button onclick="demo.nextInstance()" style="margin-left: 0.5rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border-gray); background: var(--white); border-radius: 0.25rem; cursor: pointer;">→</button>
                </div>
            ` : ''}
            <div class="comparison-container">
                <div class="comparison-column">
                    <div class="comparison-header">
                        <h3>Baseline (No Thinking)</h3>
                        <div class="budget-indicator">0 tokens</div>
                    </div>
                    ${baselinePrediction ? this.renderSinglePrediction(baselinePrediction, model, task, taskData[this.currentModel][0], true) : '<div class="no-data">No baseline data available</div>'}
                </div>
                
                <div class="comparison-column">
                    <div class="comparison-header">
                        <h3>Selected Budget</h3>
                        <div class="budget-indicator">${this.currentBudget} tokens</div>
                    </div>
                    ${this.renderSinglePrediction(currentPrediction, model, task, currentPredictions, false)}
                </div>
            </div>
        `;
    }

    renderSinglePrediction(prediction, model, task, predictions, isBaseline = false) {
        const isOSeriesModel = model.id.includes('o3') || model.id.includes('o4');
        const budget = isBaseline ? 0 : this.currentBudget;
        const hasThinking = prediction.reasoning_content && budget > 0;
        const shouldShowThinking = budget > 0;
        
        return `
            <div class="single-prediction-container">
                <div class="prediction-header-compact">
                    <div class="model-badge" style="background-color: ${model.color}">
                        ${model.name}
                    </div>
                    <div class="accuracy-badge ${prediction.correct ? 'correct' : 'incorrect'}">
                        ${prediction.correct ? '✓ Correct' : '✗ Incorrect'}
                    </div>
                </div>

                <div class="prompt-compact">
                    <div class="prompt-label">Problem</div>
                    <div class="prompt-text">${this.formatText(prediction.prompt)}</div>
                </div>

                ${shouldShowThinking ? `
                    <div class="response-section">
                        <div class="response-label">Internal Reasoning</div>
                        ${this.renderThinkingContent(prediction, isOSeriesModel, hasThinking)}
                    </div>
                ` : ''}

                <div class="response-section">
                    <div class="response-label">Final Response</div>
                    <div class="response-text">${this.highlightAnswer(prediction.response)}</div>
                </div>

                <div class="answer-comparison">
                    <strong>Expected:</strong> ${prediction.correct_answer} | 
                    <strong>Predicted:</strong> ${prediction.extracted_answer || 'N/A'}
                </div>
            </div>
        `;
    }

    renderPrediction(prediction, model, task) {
        const taskData = demoData.predictions[this.currentTask];
        const predictions = taskData[this.currentModel][this.currentBudget];
        const isOSeriesModel = model.id.includes('o3') || model.id.includes('o4');
        const hasThinking = prediction.reasoning_content && this.currentBudget > 0;
        const shouldShowThinking = this.currentBudget > 0;
        
        return `
            <div class="prediction-container">
                <div class="prediction-header">
                    <div>
                        <div class="model-badge" style="background-color: ${model.color}">
                            ${model.name}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <span style="font-size: 0.875rem; color: var(--text-secondary);">${task.description}</span>
                        </div>
                    </div>
                    <div>
                        <div class="accuracy-badge ${prediction.correct ? 'correct' : 'incorrect'}">
                            ${prediction.correct ? '✓ Correct' : '✗ Incorrect'}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-secondary); text-align: right;">
                            Budget: ${this.currentBudget === 0 ? 'No thinking' : `${this.currentBudget} tokens`}
                        </div>
                    </div>
                </div>

                <div class="prompt">
                    <div class="prompt-label">Problem Statement</div>
                    <div class="prompt-text">${this.formatText(prediction.prompt)}</div>
                </div>

                ${shouldShowThinking ? `
                    <div class="response-section">
                        <div class="response-label">Model's Internal Reasoning</div>
                        ${this.renderThinkingContent(prediction, isOSeriesModel, hasThinking)}
                    </div>
                ` : ''}

                <div class="response-section">
                    <div class="response-label">Final Model Response</div>
                    <div class="response-text">${this.highlightAnswer(prediction.response)}</div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-gray); font-size: 0.875rem; color: var(--text-secondary);">
                    <div>
                        <strong>Expected:</strong> ${prediction.correct_answer} | 
                        <strong>Predicted:</strong> ${prediction.extracted_answer || 'N/A'}
                    </div>
                    ${predictions && predictions.length > 1 ? `
                        <div>
                            <button onclick="demo.previousInstance()" style="margin-right: 0.5rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border-gray); background: var(--white); border-radius: 0.25rem; cursor: pointer;">←</button>
                            Instance ${this.currentInstance + 1} of ${predictions.length}
                            <button onclick="demo.nextInstance()" style="margin-left: 0.5rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border-gray); background: var(--white); border-radius: 0.25rem; cursor: pointer;">→</button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    renderThinkingContent(prediction, isOSeriesModel, hasThinking) {
        let content;
        if (hasThinking) {
            content = this.formatText(prediction.reasoning_content);
        } else if (isOSeriesModel) {
            content = '[Reasoning content redacted for O-series models]';
        } else {
            content = '[No internal reasoning available]';
        }
        
        return `<div class="thinking-content" style="padding: 1rem;">${content}</div>`;
    }
    

    formatText(text) {
        if (!text) return '';
        
        // Convert markdown-friendly characters to HTML
        return text
            // Convert actual newlines to line breaks
            .replace(/\n/g, '<br>')
            // Convert escaped newlines to line breaks (for string literals)
            .replace(/\\n/g, '<br>')
            // Convert \t to spaces (4 spaces for tab)
            .replace(/\\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;')
            // Convert \" to actual quotes
            .replace(/\\"/g, '"')
            // Convert \' to actual quotes
            .replace(/\\'/g, "'")
            // Escape any remaining backslashes that aren't part of markdown
            .replace(/\\\\/g, '\\');
    }

    highlightAnswer(text) {
        // First format the text, then highlight answer tags
        const formattedText = this.formatText(text);
        return formattedText.replace(/<answer>(.*?)<\/answer>/g, '<span class="answer-highlight">&lt;answer&gt;$1&lt;/answer&gt;</span>');
    }

    hidePrediction() {
        document.getElementById('predictionDisplay').style.display = 'none';
    }

    nextInstance() {
        const taskData = demoData.predictions[this.currentTask];
        const predictions = taskData[this.currentModel][this.currentBudget];
        this.currentInstance = (this.currentInstance + 1) % predictions.length;
        this.showPrediction();
    }

    previousInstance() {
        const taskData = demoData.predictions[this.currentTask];
        const predictions = taskData[this.currentModel][this.currentBudget];
        this.currentInstance = (this.currentInstance - 1 + predictions.length) % predictions.length;
        this.showPrediction();
    }

}

// Initialize the demo when the page loads
const demo = new InverseScalingDemo();

// Make demo accessible globally for button callbacks
window.demo = demo;


// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});