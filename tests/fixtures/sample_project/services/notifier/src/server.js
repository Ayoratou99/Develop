const express = require("express");
const { store } = require("./store");

const app = express();
app.use(express.json());

app.get("/notifications/:userId", (req, res) => {
  res.json(store.forUser(Number(req.params.userId)));
});

app.post("/notifications", (req, res) => {
  const { user_id, message } = req.body;
  res.status(201).json(store.add(user_id, message));
});

app.delete("/notifications/:id", (req, res) => {
  store.remove(Number(req.params.id));
  res.status(204).end();
});

app.listen(3001, () => console.log("notifier on :3001"));
