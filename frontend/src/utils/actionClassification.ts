export interface ActionGrouping {
  group1: { label: string; indices: number[]; color: string };
  group2: { label: string; indices: number[]; color: string };
  grippers: { label: string; index: number; color: string }[];
}

const COLOR_GROUP1 = "#3b82f6"; // blue
const COLOR_GROUP2 = "#8b5cf6"; // purple
const COLOR_GRIP1 = "#22c55e";  // green
const COLOR_GRIP2 = "#f97316";  // orange

export function classifyActionDimensions(
  labels: string[] | null,
  dims: number
): ActionGrouping {
  if (!labels || labels.length === 0) {
    // No labels at all — use hardcoded 7D cartesian assumption only for dims=7
    if (dims === 7) {
      return {
        group1: { label: "Position", indices: [0, 1, 2], color: COLOR_GROUP1 },
        group2: { label: "Rotation", indices: [3, 4, 5], color: COLOR_GROUP2 },
        grippers: [{ label: "Gripper", index: 6, color: COLOR_GRIP1 }],
      };
    }
    // Generic fallback — split in half, no gripper
    const mid = Math.ceil(dims / 2);
    return {
      group1: { label: `Dims (0-${mid - 1})`, indices: Array.from({ length: mid }, (_, i) => i), color: COLOR_GROUP1 },
      group2: { label: `Dims (${mid}-${dims - 1})`, indices: Array.from({ length: dims - mid }, (_, i) => mid + i), color: COLOR_GROUP2 },
      grippers: [],
    };
  }

  const lower = labels.map((l) => l.toLowerCase());

  // 1. Find gripper dimensions
  const gripperIndices: number[] = [];
  for (let i = 0; i < lower.length; i++) {
    if (lower[i].includes("gripper") || lower[i].includes("grip")) {
      gripperIndices.push(i);
    }
  }

  const nonGripperIndices = Array.from({ length: dims }, (_, i) => i).filter(
    (i) => !gripperIndices.includes(i)
  );

  // 2. Cartesian detection: look for x/y/z AND rx/ry/rz patterns
  const xIdx = lower.findIndex((l) => l === "x" || l.includes("pos_x"));
  const yIdx = lower.findIndex((l) => l === "y" || l.includes("pos_y"));
  const zIdx = lower.findIndex((l) => l === "z" || l.includes("pos_z"));
  const rxIdx = lower.findIndex((l) => l === "rx" || l.includes("rot_x") || l.includes("roll"));
  const ryIdx = lower.findIndex((l) => l === "ry" || l.includes("rot_y") || l.includes("pitch"));
  const rzIdx = lower.findIndex((l) => l === "rz" || l.includes("rot_z") || l.includes("yaw"));

  const hasCartesianPos = xIdx >= 0 && yIdx >= 0 && zIdx >= 0;
  const hasCartesianRot = rxIdx >= 0 && ryIdx >= 0 && rzIdx >= 0;

  if (hasCartesianPos && hasCartesianRot) {
    const grippers = gripperIndices.map((idx, i) => ({
      label: i === 0 ? "Gripper" : `Gripper ${i + 1}`,
      index: idx,
      color: i === 0 ? COLOR_GRIP1 : COLOR_GRIP2,
    }));
    return {
      group1: { label: "Position", indices: [xIdx, yIdx, zIdx], color: COLOR_GROUP1 },
      group2: { label: "Rotation", indices: [rxIdx, ryIdx, rzIdx], color: COLOR_GROUP2 },
      grippers,
    };
  }

  // 3. Multi-arm detection: look for left_/right_ prefixes
  const leftIndices = nonGripperIndices.filter((i) => lower[i].startsWith("left_"));
  const rightIndices = nonGripperIndices.filter((i) => lower[i].startsWith("right_"));

  if (leftIndices.length > 0 && rightIndices.length > 0) {
    const grippers = gripperIndices.map((idx) => {
      const isLeft = lower[idx].includes("left");
      return {
        label: isLeft ? "L. Grip" : "R. Grip",
        index: idx,
        color: isLeft ? COLOR_GRIP1 : COLOR_GRIP2,
      };
    });
    // Sort grippers: left first, then right
    grippers.sort((a, b) => (a.label === "L. Grip" ? -1 : 1));
    return {
      group1: { label: "Left Arm", indices: leftIndices, color: COLOR_GROUP1 },
      group2: { label: "Right Arm", indices: rightIndices, color: COLOR_GROUP2 },
      grippers,
    };
  }

  // 4. Fallback: split remaining non-gripper dims in half (larger group first)
  const mid = Math.ceil(nonGripperIndices.length / 2);
  const firstHalf = nonGripperIndices.slice(0, mid);
  const secondHalf = nonGripperIndices.slice(mid);

  const g1Start = firstHalf.length > 0 ? firstHalf[0] : 0;
  const g1End = firstHalf.length > 0 ? firstHalf[firstHalf.length - 1] : 0;
  const g2Start = secondHalf.length > 0 ? secondHalf[0] : 0;
  const g2End = secondHalf.length > 0 ? secondHalf[secondHalf.length - 1] : 0;

  const grippers = gripperIndices.map((idx, i) => ({
    label: gripperIndices.length === 1 ? "Gripper" : `Gripper ${i + 1}`,
    index: idx,
    color: i === 0 ? COLOR_GRIP1 : COLOR_GRIP2,
  }));

  return {
    group1: { label: `Joints (${g1Start}-${g1End})`, indices: firstHalf, color: COLOR_GROUP1 },
    group2: { label: `Joints (${g2Start}-${g2End})`, indices: secondHalf, color: COLOR_GROUP2 },
    grippers,
  };
}
