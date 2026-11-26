# Sculpt-Pose-Chain-Tool
A pose chain tool for Blender's sculpt mode so users can pose their meshes without leaving sculpt mode, or having to fight with the default pose brush.

**Current version: 1.6.1** · Requires Blender 4.3 or newer

# Pose-able FK Chain in Sculpt Mode
Do you ever wish that Blender's default Pose Brush tool had a bit more functionality to make posing your characters and creatures a better experience? Me too! With this N-Panel add-on you can place up to 17 points to deform your mesh without leaving Sculpt Mode!

## Installation
1. Download the latest release zip from the Releases section on the right side of this page.
2. In Blender, go to **Edit > Preferences > Get Extensions**, open the dropdown in the top-right corner, and choose **Install from Disk...**
3. Select the zip. In Sculpt Mode, press **N** and open the **Pose Chain** tab.

## How to Use
1. Set **Max Points** (3 by default, up to 17 total including branches), then click **Activate**.
2. Click on the mesh to place points. Click any previously placed point to branch new "child points" off of it, like with trees or fingers on a hand!
   - **Backspace** or **Ctrl+Z**: undo the last point
   - **Enter**: finish placing and build the chain
   - **Esc**: cancel
3. Pose the mesh:
   - Drag the first point (the anchor) to move the whole tree.
   - Drag any other point to rotate it and everything branching off of it.
   - Click empty space to keep sculpting.
   - **Ctrl+Z** undoes a pose change.
4. Click **Apply** to bake the pose into the mesh, or **Clear Points** to throw it away and restore the original mesh.

## 3 Color Coded Visuals tell you which step you're in

__Orange-red:__ Placement mode. Points are numbered in the order you place them (toggle with **Show Point Numbers**).

__Blue:__ The chain is built and ready to pose.

__Green:__ While you drag a point, it and all of its children turn green so you can see everything currently being affected.

## Weighting Controls
These are available before you activate, and they update the mesh live while a chain is active:

- **Weight Falloff**: how wide each bone's influence reaches. Raise it for thicker limbs; lower it if a bone is pulling on a neighboring area.
- **Falloff Softness**: how gradually weight fades near joints and bone tips. Low values can leave dead zones; higher values smooth things out.
- **Joint Blend**: how wide the smooth bend is at each joint. 0 is a hard crease; 1 blends across about half of the shorter bone.
- **Fit Limb Thickness**: measures how thick the mesh is around each bone so thick limbs move as a whole instead of tearing. Turn it off if a bone grabs a nearby part of the body.
- **Preserve Volume**: keeps bent joints from pinching or collapsing.

## Bend Style
Choose how the mesh bends, before activating or live while a chain is active:

- **Sharp (Bones)**: each section follows its own bone directly, for crisp joints.
- **Smooth (Lattice)**: the bones bend a lattice cage around the posed area and the mesh follows it, for softer, rounder bends with far less stretching at joints. Parts that sit closer together than a lattice cell (fingers, an arm against the body) can pull on each other a little; raise **Lattice Detail** to keep them apart.

## Fast Dragging
On by default. While you drag a point, the add-on switches to Object Mode, where Blender updates the mesh many times faster, and returns to Sculpt Mode the moment you let go. Keep it on for dense meshes, and especially with Smooth (Lattice) bends.

## Preview Bone Influence
While a chain is active, pick a point from the dropdown and click **Preview Bone Influence**. This switches to Weight Paint mode so you can see exactly which part of the mesh that bone moves and how strongly. Switch back to Sculpt Mode from the mode dropdown in the top-left of the viewport when you're done.

## Troubleshooting
Turn on **Debug Logging** to print detailed weighting info to Blender's System Console (**Window > Toggle System Console** on Windows). It's off by default for performance.

## Apply or Cancel the Tool!
Nothing is permanent until you hit Apply. It's pretty easy to test a pose, and then not even need to commit to your changes in case you need to change something before continuing the sculpt.
