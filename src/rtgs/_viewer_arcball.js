(() => {
  "use strict";

  // Viser's stock control is a world-up turntable. realtime-gs instead uses
  // IntrinsicEngine's accumulated-orientation model: each pointer delta rotates
  // around the camera's current local up/right axes, then publishes that new up
  // vector. This permits continuous motion through either pitch pole.
  const controllerKey = "__rtgsArcballCamera";
  const previousController = window[controllerKey];
  if (
    previousController !== undefined &&
    typeof previousController.dispose === "function"
  ) {
    previousController.dispose();
  }

  const config = window.__rtgsArcballConfig;
  delete window.__rtgsArcballConfig;
  if (config === undefined) {
    console.error("realtime-gs arcball camera configuration is missing");
    return;
  }

  const state = {
    disposed: false,
    retryTimer: null,
    cleanup: null,
    dispose() {
      state.disposed = true;
      if (state.retryTimer !== null) {
        window.clearTimeout(state.retryTimer);
        state.retryTimer = null;
      }
      if (state.cleanup !== null) {
        state.cleanup();
        state.cleanup = null;
      }
    },
  };
  window[controllerKey] = state;

  const install = () => {
    if (state.disposed) return;

    // Viser intentionally exposes these handles for browser integration tests.
    // Waiting also makes the persistent server message safe before React mounts.
    const mutable = window.__viserMutable;
    const controls = mutable?.cameraControl;
    const camera = mutable?.camera;
    const canvas = mutable?.canvas;
    if (
      controls === null ||
      controls === undefined ||
      camera === null ||
      camera === undefined ||
      canvas === null ||
      canvas === undefined
    ) {
      state.retryTimer = window.setTimeout(install, 25);
      return;
    }
    if (controls.mouseButtons === undefined) {
      console.error("realtime-gs arcball camera requires Viser camera-controls");
      return;
    }

    const noAction = controls.constructor.ACTION?.NONE ?? 0;
    const previousLeftAction = controls.mouseButtons.left;
    const previousOneTouchAction = controls.touches?.one;
    const previousMinDistance = controls.minDistance;
    const previousMaxDistance = controls.maxDistance;
    const previousMinPolarAngle = controls.minPolarAngle;
    const previousMaxPolarAngle = controls.maxPolarAngle;
    const previousMinAzimuthAngle = controls.minAzimuthAngle;
    const previousMaxAzimuthAngle = controls.maxAzimuthAngle;

    // Leave right-drag panning and wheel/pinch dolly with camera-controls. Only
    // rotation is replaced, so scene picking and all server camera commands stay
    // on Viser's normal path.
    controls.mouseButtons.left = noAction;
    if (controls.touches !== undefined) controls.touches.one = noAction;
    controls.minDistance = config.minDistance;
    controls.maxDistance = config.maxDistance;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;
    controls.minAzimuthAngle = -Infinity;
    controls.maxAzimuthAngle = Infinity;

    const touchPointers = new Set();
    let activePointer = null;
    let lastX = 0;
    let lastY = 0;
    let dragOrientation = null;
    let dragTarget = null;
    let dragRadius = null;

    const endDrag = () => {
      activePointer = null;
      dragOrientation = null;
      dragTarget = null;
      dragRadius = null;
    };

    const applyOrbitDelta = (xDelta, yDelta) => {
      if (
        !controls.enabled ||
        dragOrientation === null ||
        dragTarget === null ||
        dragRadius === null ||
        (xDelta === 0 && yDelta === 0)
      ) {
        return;
      }

      // Keep the gesture quaternion authoritative. Reading camera.quaternion
      // back after setLookAt() would round-trip every delta through Viser's
      // spherical controller and reintroduce its pole behavior.
      const orientation = dragOrientation;
      const right = camera.position
        .clone()
        .set(1, 0, 0)
        .applyQuaternion(orientation)
        .normalize();
      const up = camera.position
        .clone()
        .set(0, 1, 0)
        .applyQuaternion(orientation)
        .normalize();

      // Intrinsic uses +yDelta here, but its Vulkan projection explicitly flips
      // clip-space Y. Three/WebGL does not, so negate pitch to preserve the same
      // visible drag direction instead of swapping camera axes.
      const yawRotation = camera.quaternion
        .clone()
        .setFromAxisAngle(up, -xDelta * config.radiansPerPixel);
      const pitchRotation = camera.quaternion
        .clone()
        .setFromAxisAngle(right, -yDelta * config.radiansPerPixel);
      const nextOrientation = yawRotation
        .multiply(pitchRotation)
        .multiply(orientation)
        .normalize();
      dragOrientation.copy(nextOrientation);

      const forward = camera.position
        .clone()
        .set(0, 0, -1)
        .applyQuaternion(nextOrientation)
        .normalize();
      const nextUp = camera.position
        .clone()
        .set(0, 1, 0)
        .applyQuaternion(nextOrientation)
        .normalize();
      const nextPosition = dragTarget.clone().addScaledVector(forward, -dragRadius);

      // Updating camera.up before re-seeding camera-controls is the key detail:
      // the new local frame becomes the next drag delta's frame instead of being
      // projected back onto a fixed world-up turntable.
      camera.up.copy(nextUp);
      controls.updateCameraUp();
      controls.setLookAt(
        nextPosition.x,
        nextPosition.y,
        nextPosition.z,
        dragTarget.x,
        dragTarget.y,
        dragTarget.z,
        false,
      );
      controls.update(0);
      mutable.sendCamera?.();
    };

    const onPointerDown = (event) => {
      if (event.pointerType === "touch") {
        touchPointers.add(event.pointerId);
        if (touchPointers.size > 1) {
          // Two-finger pan/dolly remains owned by camera-controls.
          endDrag();
          return;
        }
      }
      if (event.button !== 0 || activePointer !== null || !controls.enabled) return;

      const target = camera.position.clone();
      controls.getTarget(target);
      const unboundedRadius = camera.position.distanceTo(target);
      if (!Number.isFinite(unboundedRadius) || unboundedRadius < 1.0e-12) return;

      dragOrientation = camera.quaternion.clone().normalize();
      dragTarget = target;
      dragRadius = Math.min(
        config.maxDistance,
        Math.max(config.minDistance, unboundedRadius),
      );
      activePointer = event.pointerId;
      lastX = event.clientX;
      lastY = event.clientY;
    };

    const onPointerMove = (event) => {
      if (event.pointerId !== activePointer) return;
      if (event.pointerType !== "touch" && (event.buttons & 1) === 0) {
        endDrag();
        return;
      }
      const xDelta = event.clientX - lastX;
      const yDelta = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      applyOrbitDelta(xDelta, yDelta);
    };

    const onPointerEnd = (event) => {
      touchPointers.delete(event.pointerId);
      if (event.pointerId === activePointer) endDrag();
    };
    const onBlur = () => {
      touchPointers.clear();
      endDrag();
    };

    canvas.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("pointermove", onPointerMove, true);
    window.addEventListener("pointerup", onPointerEnd, true);
    window.addEventListener("pointercancel", onPointerEnd, true);
    window.addEventListener("blur", onBlur);

    state.cleanup = () => {
      canvas.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("pointermove", onPointerMove, true);
      window.removeEventListener("pointerup", onPointerEnd, true);
      window.removeEventListener("pointercancel", onPointerEnd, true);
      window.removeEventListener("blur", onBlur);
      controls.mouseButtons.left = previousLeftAction;
      if (controls.touches !== undefined) controls.touches.one = previousOneTouchAction;
      controls.minDistance = previousMinDistance;
      controls.maxDistance = previousMaxDistance;
      controls.minPolarAngle = previousMinPolarAngle;
      controls.maxPolarAngle = previousMaxPolarAngle;
      controls.minAzimuthAngle = previousMinAzimuthAngle;
      controls.maxAzimuthAngle = previousMaxAzimuthAngle;
    };
  };

  install();
})();
