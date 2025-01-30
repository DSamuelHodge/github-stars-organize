export interface ChartData {
  labels: string[];
  values: number[];
}

export class ChartGenerator {
  private readonly width = 800;
  private readonly height = 400;
  private readonly padding = 50;
  private readonly barPadding = 2;
  private readonly colors = ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'];

  public generateBarChart(data: ChartData, title: string): string {
    const barWidth = (this.width - 2 * this.padding) / data.values.length - this.barPadding;
    const maxValue = Math.max(...data.values);
    const scale = (this.height - 2 * this.padding) / maxValue;

    let svg = this.initializeSVG();
    svg += this.addTitle(title);

    // Add bars
    data.values.forEach((value, i) => {
      const x = this.padding + i * (barWidth + this.barPadding);
      const barHeight = value * scale;
      const y = this.height - this.padding - barHeight;
      const color = this.colors[i % this.colors.length];

      svg += `
        <g>
          <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}"
                fill="${color}" opacity="0.8">
            <title>${data.labels[i]}: ${value}</title>
          </rect>
          <text x="${x + barWidth/2}" y="${y - 5}" 
                text-anchor="middle" font-size="12">
            ${value}
          </text>
          <text x="${x + barWidth/2}" y="${this.height - this.padding + 20}" 
                text-anchor="middle" font-size="12" transform="rotate(45, ${x + barWidth/2}, ${this.height - this.padding + 20})">
            ${data.labels[i]}
          </text>
        </g>`;
    });

    svg += this.addAxes();
    svg += '</svg>';
    return svg;
  }

  public generatePieChart(data: ChartData, title: string): string {
    const radius = Math.min(this.width, this.height) / 3;
    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const total = data.values.reduce((a, b) => a + b, 0);

    let svg = this.initializeSVG();
    svg += this.addTitle(title);

    let startAngle = 0;
    data.values.forEach((value, i) => {
      const percentage = value / total;
      const endAngle = startAngle + (percentage * Math.PI * 2);
      const color = this.colors[i % this.colors.length];

      // Calculate path
      const x1 = centerX + radius * Math.cos(startAngle);
      const y1 = centerY + radius * Math.sin(startAngle);
      const x2 = centerX + radius * Math.cos(endAngle);
      const y2 = centerY + radius * Math.sin(endAngle);
      const largeArc = percentage > 0.5 ? 1 : 0;

      svg += `
        <g>
          <path d="M ${centerX} ${centerY} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z"
                fill="${color}" opacity="0.8">
            <title>${data.labels[i]}: ${value} (${(percentage * 100).toFixed(1)}%)</title>
          </path>
        </g>`;

      // Add label
      const labelAngle = startAngle + (percentage * Math.PI);
      const labelRadius = radius * 1.2;
      const labelX = centerX + labelRadius * Math.cos(labelAngle);
      const labelY = centerY + labelRadius * Math.sin(labelAngle);
      svg += `
        <g>
          <text x="${labelX}" y="${labelY}" text-anchor="middle" font-size="12">
            ${data.labels[i]} (${(percentage * 100).toFixed(1)}%)
          </text>
        </g>`;

      startAngle = endAngle;
    });

    svg += '</svg>';
    return svg;
  }

  public generateTrendLine(data: ChartData, title: string): string {
    const maxValue = Math.max(...data.values);
    const scale = (this.height - 2 * this.padding) / maxValue;
    const step = (this.width - 2 * this.padding) / (data.values.length - 1);

    let svg = this.initializeSVG();
    svg += this.addTitle(title);

    // Generate line path
    let path = 'M ';
    data.values.forEach((value, i) => {
      const x = this.padding + i * step;
      const y = this.height - this.padding - (value * scale);
      path += `${x},${y} `;
    });

    svg += `<path d="${path}" stroke="#2563eb" stroke-width="2" fill="none"/>`;

    // Add data points and labels
    data.values.forEach((value, i) => {
      const x = this.padding + i * step;
      const y = this.height - this.padding - (value * scale);
      svg += `
        <g>
          <circle cx="${x}" cy="${y}" r="4" fill="#2563eb">
            <title>${data.labels[i]}: ${value}</title>
          </circle>
          <text x="${x}" y="${this.height - this.padding + 20}" 
                text-anchor="middle" font-size="12" transform="rotate(45, ${x}, ${this.height - this.padding + 20})">
            ${data.labels[i]}
          </text>
        </g>`;
    });

    svg += this.addAxes();
    svg += '</svg>';
    return svg;
  }

  private initializeSVG(): string {
    return `<svg width="${this.width}" height="${this.height}" xmlns="http://www.w3.org/2000/svg">
      <style>
        text { font-family: Arial, sans-serif; }
      </style>
      <rect width="${this.width}" height="${this.height}" fill="white"/>`;
  }

  private addTitle(title: string): string {
    return `
      <text x="${this.width/2}" y="${this.padding/2}" 
            text-anchor="middle" font-size="16" font-weight="bold">
        ${title}
      </text>`;
  }

  private addAxes(): string {
    return `
      <line x1="${this.padding}" y1="${this.height - this.padding}" 
            x2="${this.width - this.padding}" y2="${this.height - this.padding}" 
            stroke="black" stroke-width="1"/>
      <line x1="${this.padding}" y1="${this.height - this.padding}" 
            x2="${this.padding}" y2="${this.padding}" 
            stroke="black" stroke-width="1"/>`;
  }
}