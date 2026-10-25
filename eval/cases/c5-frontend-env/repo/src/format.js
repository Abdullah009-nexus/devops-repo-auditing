export function formatPrice(cents) {
  return `$${(cents / 100).toFixed(2)}`;
}

export const apiUrl = import.meta.env.VITE_API_URL;
