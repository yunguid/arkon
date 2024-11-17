import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import './UploadForm.css';


const cleanDescription = (description) => {
  return description
    // Convert to uppercase for consistency
    .toUpperCase()
    // Remove transaction IDs, numbers and special chars at end
    .replace(/\s*[-#\d]+\s*$/g, '')
    // Remove common payment prefixes
    .replace(/^(PAYMENT|PMT|POS|ACH|PURCHASE|DEBIT)\s*/i, '')
    // Remove dates in various formats
    .replace(/\d{1,2}\/\d{1,2}(\/\d{2,4})?/g, '')
    // Remove multiple spaces
    .replace(/\s+/g, ' ')
    // Trim whitespace
    .trim();
};

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);

  const lineChartRef = useRef();
  const barChartRef = useRef();
  const recurringChartRef = useRef();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError(null);
      console.log('Selected file:', selectedFile.name);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setError(null);
    setInsights(null);
    setSummary(null);
    setStatus('processing');

    try {
      // First request - Get immediate chart data
      const chartResponse = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
        headers: {
          Accept: 'application/json',
        },
      });

      const chartData = await chartResponse.json();
      
      if (!chartResponse.ok) {
        throw new Error(chartData.detail || `Upload failed: ${chartResponse.status}`);
      }

      // Set the summary data for immediate visualization
      setSummary(chartData.summary || null);
      
      // Comment out the second request for AI analysis
      // const analysisResponse = await fetch('http://localhost:8000/analyze', {
      //   method: 'POST',
      //   body: formData,
      //   headers: {
      //     Accept: 'application/json',
      //   },
      // });

      // const analysisData = await analysisResponse.json();
      
      // if (!analysisResponse.ok) {
      //   throw new Error(analysisData.detail || `Analysis failed: ${analysisResponse.status}`);
      // }

      // setInsights(analysisData.insights);
      setStatus('complete');
      
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
      setStatus('error');
    } finally {
      setLoading(false);
    }
  };

  const formatInsights = (insights) => {
    if (!insights) return null;
    return <div className="insights-content">{insights}</div>;
  };

  const renderCharts = (summary) => {
    if (!summary) {
      console.log('No summary data available for charts.');
      return null;
    }

    console.log('Rendering charts with summary:', summary);

    return (
      <div className="charts-container">
        <h3>Monthly Breakdown</h3>
        <div ref={lineChartRef}></div>

        <h3>Top Expenses</h3>
        <div ref={barChartRef}></div>

        <h3>Most Recurring Payments</h3>
        <div ref={recurringChartRef}></div>
      </div>
    );
  };

  useEffect(() => {
    if (summary && summary.daily_expenses) {
      renderLineChart(summary.daily_expenses);
    }
  }, [summary]);

  const renderBarChart = useCallback((data) => {
    // Remove any existing SVG
    d3.select(barChartRef.current).select('svg').remove();

    // Set up dimensions
    const margin = { top: 20, right: 30, bottom: 150, left: 70 };
    const width = 1000 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;

    // Create the SVG element
    const svg = d3
      .select(barChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Process data to aggregate by description
    const topData = processTransactions(data);

    // Set up scales
    const x = d3
      .scaleBand()
      .domain(topData.map((d) => d.description))
      .range([0, width])
      .padding(0.2);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(topData, (d) => d.totalAmount)])
      .nice()
      .range([height, 0]);

    // Add X axis with wrapped text
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    // Remove the default text
    xAxis.selectAll('text').remove();

    // Add wrapped text labels
    xAxis.selectAll('g.tick')
      .append('foreignObject')
      .attr('x', -40)  // Adjust position
      .attr('y', 0)    // Adjust position
      .attr('width', 80)  // Fixed width for text container
      .attr('height', 70) // Fixed height for text container
      .append('xhtml:div')
      .style('text-align', 'center')
      .style('font-size', '11px')
      .style('color', '#1a237e')
      .style('word-wrap', 'break-word')
      .style('display', 'flex')
      .style('align-items', 'center')
      .style('justify-content', 'center')
      .style('height', '100%')
      .text(d => d);

    // Add Y axis with currency format
    g.append('g')
      .call(d3.axisLeft(y).tickFormat(d => `$${d3.format(",.0f")(d)}`))
      .selectAll('text')
      .style('fill', '#1a237e');

    // Add gridlines
    g.append('g')
      .attr('class', 'grid')
      .call(
        d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
      )
      .selectAll('line')
      .style('stroke', 'rgba(26, 35, 126, 0.1)');

    // Create the bars
    g.selectAll('.bar')
      .data(topData)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (d) => x(d.description))
      .attr('y', (d) => y(d.totalAmount))
      .attr('width', x.bandwidth())
      .attr('height', (d) => height - y(d.totalAmount))
      .attr('fill', '#1a237e');

    // Add count labels if more than 1 occurrence
    g.selectAll('.label')
      .data(topData)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', (d) => x(d.description) + x.bandwidth() / 2)
      .attr('y', (d) => y(d.totalAmount) - 5)
      .attr('text-anchor', 'middle')
      .style('fill', '#1a237e')
      .style('font-size', '12px')
      .text((d) => d.count > 1 ? `x${d.count}` : '');

    // Add tooltips
    const tooltip = d3
      .select(barChartRef.current)
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0);

    g.selectAll('.bar')
      .on('mouseover', (event, d) => {
        tooltip.transition().duration(200).style('opacity', 0.9);
        tooltip
          .html(
            `<strong>${d.description}</strong><br/>Total Amount: $${d.totalAmount.toFixed(
              2
            )}<br/>Occurrences: ${d.count}`
          )
          .style('left', event.pageX + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', () => {
        tooltip.transition().duration(500).style('opacity', 0);
      });
  }, []);

  useEffect(() => {
    if (summary && summary.expenses_by_description) {
      renderBarChart(summary.expenses_by_description);
    }
  }, [summary, renderBarChart]);

  const renderLineChart = (data) => {
    // Remove any existing SVG
    d3.select(lineChartRef.current).select('svg').remove();

    // Set up the SVG canvas dimensions
    const margin = { top: 20, right: 20, bottom: 70, left: 70 };
    const width = 1000 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    // Create the SVG element
    const svg = d3
      .select(lineChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Parse the date
    const parseDate = d3.timeParse('%Y-%m-%d');

    // Format the data
    const formattedData = data.map((d) => ({
      date: parseDate(d.Date),
      amount: +d.Amount,
    }));

    // Set up scales
    const x = d3
      .scaleTime()
      .domain(d3.extent(formattedData, (d) => d.date))
      .range([0, width]);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(formattedData, (d) => d.amount) * 1.1])
      .range([height, 0]);

    // Calculate number of days between start and end
    const startDate = d3.min(formattedData, d => d.date);
    const endDate = d3.max(formattedData, d => d.date);
    const days = d3.timeDay.count(startDate, endDate);

    // Calculate tick values manually to ensure even spacing
    const tickValues = [];
    let currentDate = new Date(startDate);
    for (let i = 0; i <= days; i += 2) { // Every 2 days
      tickValues.push(new Date(currentDate));
      currentDate.setDate(currentDate.getDate() + 2);
    }

    // Update the X axis configuration
    const xAxis = d3.axisBottom(x)
      .tickValues(tickValues)
      .tickFormat(d3.timeFormat('%b %d'))
      .tickSizeOuter(0);

    // Update x scale domain
    x.domain([startDate, endDate]);

    // Update the Y axis to show currency format
    const yAxis = d3.axisLeft(y)
      .tickFormat(d => `$${d3.format(",.0f")(d)}`); // Format as "$1,000"

    // Update the axis styling and labels
    // After adding X axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')
      .style('fill', '#1a237e')
      .style('font-size', '12px');

    // Add X axis label
    g.append('text')
      .attr('transform', `translate(${width/2}, ${height + margin.bottom - 10})`)
      .style('text-anchor', 'middle')
      .style('fill', '#1a237e')
      .style('font-size', '14px')
      .text('Date');

    // Update Y axis
    g.append('g')
      .call(yAxis)
      .selectAll('text')
      .style('fill', '#1a237e')
      .style('font-size', '12px');

    // Add Y axis label with adjusted positioning
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left - 5)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('fill', '#1a237e')
      .style('font-size', '14px')
      .text('Amount ($)');

    // Update gridlines to be more subtle
    g.append('g')
      .attr('class', 'grid')
      .call(
        d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
      )
      .selectAll('line')
      .style('stroke', 'rgba(26, 35, 126, 0.1)');

    // Add the line
    g.append('path')
      .datum(formattedData)
      .attr('fill', 'none')
      .attr('stroke', '#1a237e')
      .attr('stroke-width', 2)
      .attr(
        'd',
        d3
          .line()
          .x((d) => x(d.date))
          .y((d) => y(d.amount))
      );

    // Add dots
    g.selectAll('dot')
      .data(formattedData)
      .enter()
      .append('circle')
      .attr('cx', (d) => x(d.date))
      .attr('cy', (d) => y(d.amount))
      .attr('r', 4)
      .attr('fill', '#1a237e');

    // Add Tooltip
    const tooltip = d3
      .select(lineChartRef.current)
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0);

    g.selectAll('circle')
      .on('mouseover', (event, d) => {
        tooltip.transition().duration(200).style('opacity', 0.9);
        tooltip
          .html(
            `<strong>Date:</strong> ${d3.timeFormat('%b %d, %Y')(d.date)}<br/><strong>Amount:</strong> $${d.amount.toFixed(
              2
            )}`
          )
          .style('left', event.pageX + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', () => {
        tooltip.transition().duration(500).style('opacity', 0);
      });
  };

  const processTransactions = (data) => {
    const paymentsMap = new Map();
    
    data.forEach(d => {
      const description = cleanDescription(d.Description);
      const amount = Math.abs(+d.Amount); // Use absolute value

      if (!description) return; // Skip empty descriptions
      
      const existing = paymentsMap.get(description) || { totalAmount: 0, count: 0 };
      paymentsMap.set(description, {
        description,
        totalAmount: existing.totalAmount + amount,
        count: existing.count + 1
      });
    });

    return Array.from(paymentsMap.values())
      .sort((a, b) => b.totalAmount - a.totalAmount)
      .slice(0, 10); // Top 10 expenses
  };

  const renderRecurringChart = useCallback((data) => {
    console.log('Rendering recurring chart with data:', data);
    
    // Remove existing SVG
    d3.select(recurringChartRef.current).select('svg').remove();

    if (!data || !data.length) {
        console.log('No recurring payments found');
        return;
    }

    // Process the data to ensure correct property access
    const recurringPayments = data.map(d => ({
        description: d.Description || '',  // Use Description instead of description
        count: d.count || 0,
        totalAmount: d.totalAmount || 0,
        averageAmount: d.averageAmount || 0
    }));

    console.log('Processed recurring payments:', recurringPayments);

    // Set up dimensions
    const margin = { top: 20, right: 30, bottom: 150, left: 70 };
    const width = 1000 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(recurringChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Set up scales with correct property access
    const x = d3.scaleBand()
      .domain(recurringPayments.map(d => d.description))
      .range([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .domain([0, d3.max(recurringPayments, d => d.count)])
      .nice()
      .range([height, 0]);

    // Add X axis with wrapped text
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    xAxis.selectAll('text').remove();

    // Add wrapped text labels
    xAxis.selectAll('g.tick')
      .append('foreignObject')
      .attr('x', -40)
      .attr('y', 0)
      .attr('width', 80)
      .attr('height', 70)
      .append('xhtml:div')
      .style('text-align', 'center')
      .style('font-size', '11px')
      .style('color', '#1a237e')
      .style('word-wrap', 'break-word')
      .style('display', 'flex')
      .style('align-items', 'center')
      .style('justify-content', 'center')
      .style('height', '100%')
      .text(d => cleanDescription(d)); // Clean the description for display

    // Add Y axis
    g.append('g')
      .call(d3.axisLeft(y).tickFormat(d => Math.round(d))) // Show whole numbers
      .selectAll('text')
      .style('fill', '#1a237e');

    // Add bars with correct property access
    const bars = g.selectAll('.bar')
      .data(recurringPayments)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.description))
      .attr('y', d => y(d.count))
      .attr('width', x.bandwidth())
      .attr('height', d => height - y(d.count))
      .attr('fill', '#4CAF50');

    // Update tooltip with correct property access
    const tooltip = d3.select(recurringChartRef.current)
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0);

    bars.on('mouseover', (event, d) => {
      tooltip.transition()
        .duration(200)
        .style('opacity', 0.9);
      tooltip.html(
        `<strong>${cleanDescription(d.description)}</strong><br/>
         Times per month: ${d.count}<br/>
         Average amount: $${d.averageAmount.toFixed(2)}`
      )
        .style('left', event.pageX + 'px')
        .style('top', event.pageY - 28 + 'px');
    })
      .on('mouseout', () => {
        tooltip.transition()
          .duration(500)
          .style('opacity', 0);
      });
  }, []);

  useEffect(() => {
    if (summary && summary.recurring_transactions) {
        renderRecurringChart(summary.recurring_transactions);
    }
  }, [summary, renderRecurringChart]);

  return (
    <div className="upload-container">
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-wrapper">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="file-input"
            id="file-input"
          />
          <label htmlFor="file-input" className="file-input-label">
            {fileName || 'Choose a CSV file to analyze'}
          </label>
        </div>
        <button
          type="submit"
          className="upload-button"
          disabled={!file || loading}
        >
          {loading ? 'Processing...' : 'Analyze Data'}
        </button>
      </form>

      {status === 'processing' && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <span>Processing your financial data...</span>
        </div>
      )}

      {status === 'complete' && (
        <>
          {summary && renderCharts(summary)}
          {insights && (
            <div className="insights-container">{formatInsights(insights)}</div>
          )}
        </>
      )}

      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default UploadForm;
