// differential-helper.js

const DifferentialDiagnosis = {
    render: function(container, data, onChange) {
        // Create root element if it doesn't exist
        const root = document.getElementById('differential-root');
        if (!root) {
            console.error('Could not find root element');
            return;
        }

        // Initialize the React component
        const Component = window.DifferentialDiagnosis;
        if (!Component) {
            console.error('React component not found');
            return;
        }

        // Render the component
        ReactDOM.render(
            React.createElement(Component, {
                initialDiagnoses: data,
                onChange: onChange
            }),
            container
        );
    }
};

// Attach to window for global access
window.DifferentialDiagnosis = DifferentialDiagnosis;
