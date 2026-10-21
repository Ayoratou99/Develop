let seq = 1;
const items = [];

function add(userId, message) {
  const n = { id: seq++, user_id: userId, message, created_at: new Date().toISOString() };
  items.push(n);
  return n;
}

function forUser(userId) {
  return items.filter((n) => n.user_id === userId);
}

function remove(id) {
  const i = items.findIndex((n) => n.id === id);
  if (i >= 0) items.splice(i, 1);
}

module.exports = { store: { add, forUser, remove } };
