import { DEMO_DATA } from './demo_data.js';

// Simple script for anonymous demo with side-by-side display

class InverseScalingDemo {
    constructor() {
        this.currentTask = null;
        this.currentModel = null;
        this.currentInstance = null;
        this.currentBudget = 0;
        this.demoData = DEMO_DATA;
        
        // Wait for demo data to be loaded
        if (this.demoData) {
            this.init();
        } else {
            window.addEventListener('load', () => this.init());
        }
    }

    init() {
        this.populateTaskSelect();
        this.setupEventListeners();
    }

    populateTaskSelect() {
        const taskSelect = document.getElementById('taskSelect');
        taskSelect.innerHTML = '<option value="">Select a task...</option>';
        
        this.demoData.tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task.id;
            option.textContent = task.name;
            taskSelect.appendChild(option);
        });
    }

    populateModelSelect() {
        const modelSelect = document.getElementById('modelSelect');
        if (!this.currentTask || !this.demoData.predictions[this.currentTask]) {
            modelSelect.innerHTML = '<option value="">Select a task first</option>';
            return;
        }
        
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        
        const availableModels = Object.keys(this.demoData.predictions[this.currentTask]);
        
        this.demoData.models.forEach(model => {
            if (availableModels.includes(model.id)) {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                modelSelect.appendChild(option);
            }
        });
    }

    populateInstanceSelect() {
        const instanceSelect = document.getElementById('instanceSelect');
        if (!this.currentTask || !this.currentModel || !this.demoData.predictions[this.currentTask][this.currentModel]) {
            instanceSelect.innerHTML = '<option value="">Select a model first</option>';
            return;
        }
        
        instanceSelect.innerHTML = '<option value="">Select an instance...</option>';
        
        const predictionsForModel = this.demoData.predictions[this.currentTask][this.currentModel];
        const instances = new Set();
        
        Object.values(predictionsForModel).forEach(budgetPredictions => {
            budgetPredictions.forEach(p => instances.add(p.instance_id))
        });

        Array.from(instances).forEach((instanceId, index) => {
            const option = document.createElement('option');
            option.value = instanceId;
            option.textContent = `Example ${index + 1}`;
            instanceSelect.appendChild(option);
        });
    }

    setupEventListeners() {
        document.getElementById('taskSelect').addEventListener('change', (e) => {
            this.currentTask = e.target.value;
            this.currentModel = null;
            this.currentInstance = null;
            this.populateModelSelect();
            this.populateInstanceSelect();
            this.clearDisplay();
        });
        
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            this.currentModel = e.target.value;
            this.currentInstance = null;
            this.populateInstanceSelect();
            this.clearDisplay();
        });
        
        document.getElementById('instanceSelect').addEventListener('change', (e) => {
            this.currentInstance = e.target.value;
            if (this.currentInstance) {
                this.displayInstance();
            } else {
                this.clearDisplay();
            }
        });
    }

    getPredictionsForInstance() {
        if (!this.currentTask || !this.currentModel || !this.currentInstance) return [];

        const predictionsForModel = this.demoData.predictions[this.currentTask][this.currentModel];
        const instancePredictions = [];
        
        Object.entries(predictionsForModel).forEach(([budget, budgetPredictions]) => {
            const pred = budgetPredictions.find(p => p.instance_id === this.currentInstance);
            if (pred) {
                instancePredictions.push(pred);
            }
        });
        return instancePredictions.sort((a, b) => a.reasoning_budget - b.reasoning_budget);
    }

    displayInstance() {
        const predictions = this.getPredictionsForInstance();
        if (predictions.length === 0) return;
        
        this.createBudgetTabs(predictions);
        
        this.currentBudget = predictions.find(p => p.reasoning_budget > 0)?.reasoning_budget ?? 0;
        if (this.currentBudget === 0 && predictions.length > 1) {
            this.currentBudget = predictions[1].reasoning_budget;
        } else if (predictions.length > 0 && this.currentBudget === 0) {
            this.currentBudget = predictions[0].reasoning_budget;
        }

        const activeTab = document.querySelector(`.budget-tab[data-budget='${this.currentBudget}']`);
        if(activeTab) {
            document.querySelectorAll('.budget-tab').forEach(t => t.classList.remove('active'));
            activeTab.classList.add('active');
        }

        this.displayComparison();
    }

    createBudgetTabs(predictions) {
        const budgetTabs = document.getElementById('budgetTabs');
        budgetTabs.innerHTML = '';
        budgetTabs.style.display = 'flex';
        
        predictions.forEach((pred) => {
            const tab = document.createElement('button');
            tab.className = 'budget-tab';
            tab.textContent = `Budget: ${pred.reasoning_budget}`;
            tab.dataset.budget = pred.reasoning_budget;
            
            if (pred.reasoning_budget === this.currentBudget) {
                 tab.classList.add('active');
            }
            
            tab.addEventListener('click', () => {
                document.querySelectorAll('.budget-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.currentBudget = pred.reasoning_budget;
                this.displayComparison();
            });
            
            budgetTabs.appendChild(tab);
        });
    }

    displayComparison() {
        const predictions = this.getPredictionsForInstance();
        const baselinePrediction = predictions.find(p => p.reasoning_budget === 0);
        const currentPrediction = predictions.find(p => p.reasoning_budget === this.currentBudget);
        
        if (!currentPrediction) return;
        
        const display = document.getElementById('predictionDisplay');
        
        const comparisonHtml = `
            <div class="comparison-container">
                <div class="comparison-column">
                    <div class="comparison-header">
                        <h3>Baseline (No Thinking)</h3>
                        <div class="budget-indicator">0 tokens</div>
                    </div>
                    ${baselinePrediction ? this.renderSinglePrediction(baselinePrediction, true) : '<div class="no-data">No baseline data available</div>'}
                </div>
                
                <div class="comparison-column">
                    <div class="comparison-header">
                        <h3>Selected Budget</h3>
                        <div class="budget-indicator">${this.currentBudget} tokens</div>
                    </div>
                    ${this.renderSinglePrediction(currentPrediction, false)}
                </div>
            </div>
        `;
        
        display.innerHTML = comparisonHtml;
    }

    renderSinglePrediction(prediction, isBaseline = false) {
        const model = this.demoData.models.find(m => m.id === this.currentModel)
        const modelName = model ? model.name : this.currentModel;
        const modelColor = this.getModelColor(this.currentModel);
        const hasThinking = prediction.reasoning_content && !isBaseline;
        
        return `
            <div class="single-prediction-container">
                <div class="prediction-header-compact">
                    <div class="model-badge" style="background-color: ${modelColor}">
                        ${modelName}
                    </div>
                    <div class="accuracy-badge ${prediction.correct ? 'correct' : 'incorrect'}">
                        ${prediction.correct ? '✓ Correct' : '✗ Incorrect'}
                    </div>
                </div>

                <div class="prompt-compact">
                    <div class="prompt-label">Problem</div>
                    <div class="prompt-text">${this.formatText(prediction.prompt)}</div>
                </div>

                ${hasThinking ? `
                    <div class="response-section">
                        <div class="response-label">Internal Reasoning</div>
                        <div class="thinking-text">
                            ${this.formatText(prediction.reasoning_content)}
                        </div>
                    </div>
                ` : ''}

                <div class="response-section">
                    <div class="response-label">Final Response</div>
                    <div class="response-text">${this.formatResponse(prediction.response)}</div>
                </div>

                <div class="answer-display">
                    <div class="answer-item">
                        <span class="answer-label">Given Answer</span>
                        <span class="answer-value">${this.escapeHtml(String(prediction.extracted_answer || 'N/A'))}</span>
                    </div>
                    <div class="answer-item">
                        <span class="answer-label">Expected Answer</span>
                        <span class="answer-value">${this.escapeHtml(String(prediction.correct_answer || 'N/A'))}</span>
                    </div>
                </div>
            </div>
        `;
    }

    clearDisplay() {
        document.getElementById('budgetTabs').style.display = 'none';
        document.getElementById('predictionDisplay').innerHTML = '';
    }

    getModelDisplayName(modelId) {
        const model = this.demoData.models.find(m => m.id === modelId);
        return model ? model.name : modelId;
    }
    
    getModelColor(modelId) {
        const colors = {
            'claude-sonnet-4-20250514': '#f97316',
            'claude-opus-4-20250514': '#ea580c',
            'claude-3-7-sonnet-20250219': '#fb923c',
            'deepseek_r1_0528_awq': '#22c55e'
        };
        const defaultColor = '#6b7280';
        
        const model = this.demoData.models.find(m => m.id === modelId);
        if(model) {
            if (model.name.toLowerCase().includes('opus')) return colors['claude-opus-4-20250514'];
            if (model.name.toLowerCase().includes('sonnet')) return colors['claude-sonnet-4-20250514'];
            if (model.name.toLowerCase().includes('deepseek')) return colors['deepseek_r1_0528_awq'];
        }
        return defaultColor;
    }

    formatText(text) {
        if (typeof text !== 'string') {
            return '';
        }
        return this.escapeHtml(text).replace(/\n/g, '<br>');
    }
    
    formatResponse(response) {
        let formatted = this.formatText(response);
        // Highlight <answer> tags
        formatted = formatted.replace(/&lt;answer&gt;(.*?)&lt;\/answer&gt;/, '<code class="answer-tag">&lt;answer&gt;$1&lt;/answer&gt;</code>');
        return formatted;
    }

    escapeHtml(text) {
        if (text === null || text === undefined) {
            return '';
        }
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const demo = new InverseScalingDemo();
});