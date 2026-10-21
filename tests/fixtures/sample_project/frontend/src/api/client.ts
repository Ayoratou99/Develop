import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

export const http = axios.create({ baseURL: API_BASE });

export interface Order {
  id: number;
  reference: string;
  customer_id: number;
  status: "draft" | "confirmed" | "shipped" | "cancelled";
  total_amount: number;
  created_at: string;
}

export interface Customer {
  id: number;
  email: string;
  full_name: string;
}

export async function listOrders(): Promise<Order[]> {
  const res = await http.get<Order[]>("/orders");
  return res.data;
}

export async function getOrder(id: number): Promise<Order> {
  const res = await http.get<Order>(`/orders/${id}`);
  return res.data;
}

export async function confirmOrder(id: number): Promise<Order> {
  const res = await http.post<Order>(`/orders/${id}/confirm`);
  return res.data;
}

export async function searchCustomers(q: string): Promise<Customer[]> {
  const res = await fetch(`${API_BASE}/customers?q=${encodeURIComponent(q)}`);
  return res.json();
}
