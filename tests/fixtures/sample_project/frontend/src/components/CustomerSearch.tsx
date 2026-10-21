import React, { useState } from "react";
import { searchCustomers, Customer } from "../api/client";

export default function CustomerSearch() {
  const [results, setResults] = useState<Customer[]>([]);
  return (
    <div>
      <input onChange={(e) => searchCustomers(e.target.value).then(setResults)} />
      <ul>{results.map((c) => <li key={c.id}>{c.full_name} ({c.email})</li>)}</ul>
    </div>
  );
}
