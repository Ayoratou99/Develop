import { useEffect, useState } from "react";

export function Notifications({ userId }) {
  const [items, setItems] = useState([]);
  useEffect(() => {
    fetch("http://localhost:3001/notifications/" + userId).then((r) => r.json()).then(setItems);
  }, [userId]);
  return <ul>{items.map((n) => <li key={n.id}>{n.message}</li>)}</ul>;
}
