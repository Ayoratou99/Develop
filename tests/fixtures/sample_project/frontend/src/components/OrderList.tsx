import React, { useEffect, useState } from "react";
import { listOrders, confirmOrder, Order } from "../api/client";
import { formatAmount } from "../utils/money";

export function OrderList() {
  const [orders, setOrders] = useState<Order[]>([]);
  useEffect(() => { listOrders().then(setOrders); }, []);
  return (
    <table>
      {orders.map((o) => (
        <tr key={o.id}>
          <td>{o.reference}</td>
          <td>{o.status}</td>
          <td>{formatAmount(o.total_amount)}</td>
          <td><button onClick={() => confirmOrder(o.id)}>Confirmer</button></td>
        </tr>
      ))}
    </table>
  );
}
