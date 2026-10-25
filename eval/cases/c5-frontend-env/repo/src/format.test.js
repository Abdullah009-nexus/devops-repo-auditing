import { expect, test } from "vitest";
import { formatPrice } from "./format";

test("formats cents as dollars", () => {
  expect(formatPrice(1250)).toBe("$12.50");
});
