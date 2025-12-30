/**
 * Frontend Streaming Client for Agentic Workflows
 *
 * Provides easy-to-use API for consuming SSE streams from Flask backend
 */

class AgenticStreamingClient {
    constructor(baseURL = '/api/agentic') {
        this.baseURL = baseURL;
    }

    /**
     * Call a streaming workflow endpoint
     *
     * @param {string} endpoint - Endpoint path (e.g., '/solution/stream')
     * @param {object} params - Request parameters
     * @param {function} onProgress - Callback for progress updates
     * @param {function} onComplete - Callback for completion
     * @param {function} onError - Callback for errors
     * @returns {object} Controller to abort the stream
     */
    async callStream(endpoint, params, onProgress, onComplete, onError) {
        const controller = new AbortController();
        const url = this.baseURL + endpoint;

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include', // Include session cookie
                body: JSON.stringify(params),
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                // Decode chunk
                buffer += decoder.decode(value, { stream: true });

                // Process complete lines
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep last incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            this._handleProgress(data, onProgress, onComplete, onError);
                        } catch (e) {
                            console.error('Failed to parse SSE data:', e, line);
                        }
                    }
                }
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Stream aborted');
            } else {
                console.error('Stream error:', error);
                if (onError) {
                    onError(error);
                }
            }
        }

        return {
            abort: () => controller.abort()
        };
    }

    _handleProgress(data, onProgress, onComplete, onError) {
        if (data.error) {
            if (onError) {
                onError(data);
            }
        } else if (data.step === 'complete') {
            if (onComplete) {
                onComplete(data);
            }
        } else {
            if (onProgress) {
                onProgress(data);
            }
        }
    }

    /**
     * Call Solution Workflow with streaming
     */
    async solutionWorkflow(message, sessionId, callbacks) {
        return this.callStream(
            '/solution/stream',
            { message, session_id: sessionId },
            callbacks.onProgress,
            callbacks.onComplete,
            callbacks.onError
        );
    }

    /**
     * Call Comparison Workflow with streaming
     */
    async comparisonWorkflow(message, sessionId, callbacks) {
        return this.callStream(
            '/compare/stream',
            { message, session_id: sessionId },
            callbacks.onProgress,
            callbacks.onComplete,
            callbacks.onError
        );
    }

    /**
     * Call Comparison from Spec with streaming
     */
    async compareFromSpec(specObject, comparisonType, sessionId, callbacks) {
        return this.callStream(
            '/compare-from-spec/stream',
            { spec_object: specObject, comparison_type: comparisonType, session_id: sessionId },
            callbacks.onProgress,
            callbacks.onComplete,
            callbacks.onError
        );
    }

    /**
     * Call Instrument Detail Workflow with streaming
     */
    async instrumentDetailWorkflow(message, sessionId, callbacks) {
        return this.callStream(
            '/instrument-detail/stream',
            { message, session_id: sessionId },
            callbacks.onProgress,
            callbacks.onComplete,
            callbacks.onError
        );
    }

    /**
     * Call Grounded Chat with streaming
     */
    async groundedChat(question, sessionId, callbacks) {
        return this.callStream(
            '/chat-knowledge/stream',
            { question, session_id: sessionId },
            callbacks.onProgress,
            callbacks.onComplete,
            callbacks.onError
        );
    }

    /**
     * Call Smart Chat (auto-routing) with streaming
     */
    async smartChat(message, sessionId, callbacks) {
        return this.callStream(
            '/smart-chat/stream',
            { message, session_id: sessionId },
            callbacks.onProgress,
            callbacks.onComplete,
            callbacks.onError
        );
    }
}


// ============================================================================
// USAGE EXAMPLES
// ============================================================================

/**
 * Example 1: Basic usage with Solution Workflow
 */
function exampleSolutionWorkflow() {
    const client = new AgenticStreamingClient();

    const controller = client.solutionWorkflow(
        "Need SIL2 pressure transmitters for crude oil refinery",
        null, // session_id (null = auto-generate)
        {
            onProgress: (data) => {
                console.log(`[${data.step}] ${data.message} - ${data.progress}%`);
                updateProgressBar(data.progress);
                addStatusMessage(data.message);
            },
            onComplete: (data) => {
                console.log('Workflow complete!', data);
                displayResults(data.data);
            },
            onError: (error) => {
                console.error('Workflow error:', error);
                showError(error.message);
            }
        }
    );

    // Can abort the stream if needed
    // controller.then(c => c.abort());
}


/**
 * Example 2: Smart Chat with routing
 */
function exampleSmartChat() {
    const client = new AgenticStreamingClient();

    client.smartChat(
        "What is the difference between SIL2 and SIL3?",
        null,
        {
            onProgress: (data) => {
                // Handle routing info
                if (data.step === 'routed') {
                    console.log(`Routed to: ${data.data.workflow}`);
                    showRoutingInfo(data.data.workflow, data.data.confidence);
                }

                // Update progress
                updateProgressBar(data.progress);
                addStatusMessage(data.message);
            },
            onComplete: (data) => {
                if (data.data.workflow === 'grounded_chat') {
                    displayAnswer(data.data.answer, data.data.citations);
                } else if (data.data.workflow === 'solution') {
                    displayProductResults(data.data.ranked_results);
                }
            },
            onError: (error) => {
                showError(error.message);
            }
        }
    );
}


/**
 * Example 3: Comparison from Spec
 */
function exampleCompareFromSpec() {
    const client = new AgenticStreamingClient();

    const specObject = {
        product_type: "pressure transmitter",
        specifications: {
            range: "0-500 psi",
            accuracy: "0.04%",
            outputSignal: "4-20mA"
        },
        required_certifications: ["SIL2", "ATEX"]
    };

    client.compareFromSpec(
        specObject,
        'full', // comparison_type
        null,
        {
            onProgress: (data) => {
                console.log(`[${data.step}] ${data.progress}%`);

                // Show which comparison level is running
                if (data.step === 'vendor_comparison') {
                    showComparisonStage('Comparing Vendors');
                } else if (data.step === 'series_comparison') {
                    showComparisonStage('Comparing Product Series');
                } else if (data.step === 'model_comparison') {
                    showComparisonStage('Comparing Models');
                }

                updateProgressBar(data.progress);
            },
            onComplete: (data) => {
                displayComparisonResults(data.data);
            },
            onError: (error) => {
                showError(error.message);
            }
        }
    );
}


// ============================================================================
// UI HELPER FUNCTIONS (implement these in your app)
// ============================================================================

function updateProgressBar(progress) {
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';
    }
}

function addStatusMessage(message) {
    const statusContainer = document.getElementById('status-messages');
    if (statusContainer) {
        const messageEl = document.createElement('div');
        messageEl.className = 'status-message';
        messageEl.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
        statusContainer.appendChild(messageEl);
        statusContainer.scrollTop = statusContainer.scrollHeight;
    }
}

function displayResults(results) {
    console.log('Display results:', results);
    // Implement your result display logic
}

function displayAnswer(answer, citations) {
    console.log('Display answer:', answer, citations);
    // Implement your answer display logic
}

function displayProductResults(rankedResults) {
    console.log('Display products:', rankedResults);
    // Implement your product display logic
}

function displayComparisonResults(comparisonData) {
    console.log('Display comparison:', comparisonData);
    // Implement your comparison display logic
}

function showError(message) {
    console.error('Error:', message);
    alert('Error: ' + message);
}

function showRoutingInfo(workflow, confidence) {
    console.log(`Routing to ${workflow} (confidence: ${confidence})`);
}

function showComparisonStage(stage) {
    console.log(`Comparison stage: ${stage}`);
}


// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AgenticStreamingClient;
}
