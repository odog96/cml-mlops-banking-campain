/**
 * Module 3: Proactive MLOps - Data Drift Monitoring Dashboard
 * Simplified JavaScript for displaying drift detection results
 */

document.addEventListener('DOMContentLoaded', function() {
    loadDriftStatus();
});

/**
 * Fetch and display the latest drift status
 */
function loadDriftStatus() {
    axios.get('/drift-status')
        .then(function(response) {
            const data = response.data;
            const alertElement = document.getElementById('driftStatusAlert');

            if (data.status === 'FAIL') {
                alertElement.className = 'alert alert-danger';
                alertElement.innerHTML = `
                    <strong>⚠️ Data Drift Detected!</strong><br>
                    Status: <span class="badge bg-danger">${data.status}</span><br>
                    Tests: ${data.total_tests} total,
                    <span class="badge bg-success">${data.passed_tests} passed</span>,
                    <span class="badge bg-danger">${data.failed_tests} failed</span><br>
                    <small class="text-muted">Timestamp: ${new Date(data.timestamp).toLocaleString()}</small>
                `;
            } else if (data.status === 'PASS') {
                alertElement.className = 'alert alert-success';
                alertElement.innerHTML = `
                    <strong>✓ No Data Drift Detected</strong><br>
                    Status: <span class="badge bg-success">${data.status}</span><br>
                    Tests: ${data.total_tests} total,
                    <span class="badge bg-success">${data.passed_tests} passed</span><br>
                    <small class="text-muted">Timestamp: ${new Date(data.timestamp).toLocaleString()}</small>
                `;
            } else {
                alertElement.className = 'alert alert-warning';
                alertElement.innerHTML = `
                    <strong>ℹ️ Status Unknown</strong><br>
                    ${data.message || 'No drift status data available yet.'}
                `;
            }
        })
        .catch(function(error) {
            const alertElement = document.getElementById('driftStatusAlert');
            alertElement.className = 'alert alert-warning';
            alertElement.innerHTML = `
                <strong>ℹ️ Drift Report Not Available</strong><br>
                ${error.response?.data?.message || 'Please run the drift detection script (1_check_drift.py) first.'}
            `;
        });
}
