import { render, screen } from '@testing-library/react';
import App from '../src/App';

test('renders upload button', () => {
  render(<App />);
  const buttonElement = screen.getByText(/Upload and Analyze/i);
  expect(buttonElement).toBeInTheDocument();
}); 