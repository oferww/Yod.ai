export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface Product {
  SKU: string;
  Brand: string;
  Family: string;
  Name: string;
  Description: string;
  CPU?: string;
  GPU?: string;
  Storage?: string;
  RAM?: string;
  Price?: string;
}

export interface ChatResponse {
  role: 'assistant';
  content: string;
}
