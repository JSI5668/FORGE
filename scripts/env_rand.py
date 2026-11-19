"""Appearance domain randomization: per episode, randomize the colors of the warehouse floor, walls, shelves,
pallets and boxes, and the light color.

- The original appearance (unchanged textures) is kept with probability p_orig (default 0.25)
- Some colors are excluded from training so they do not overlap the held-out "unseen warehouse" test variants 1 and 2:
    green floors (variant 1 = green epoxy), blue pallets (variant 1 = blue plastic),
    blue shelves and pillars (variant 2 = blue shelves)
  -> evaluation on variants 1 and 2 remains a generalization test on unseen colors
- The scene file is not modified (material bindings are changed at runtime only). Geometry (positions, sizes) is unchanged
Usage: R = EnvRand(stage, log); R.episode(rng)  # at the start of every episode
"""
import colorsys

# material name -> category
CATS = {
    "floor": ["MI_Floor_01"],
    "stripe": ["M_FloorStripes_01"],
    "wall": ["MI_WallA_01", "MI_WallB_01"],
    "wall_detail": ["MI_WallDetails_01"],
    "steel": ["MI_BeamsA_01", "MI_BeamsA_02", "MI_FrameA_01",
              "MI_RackSetA_01", "MI_RackSetB_01", "MI_RackSetC_01", "MI_RackSetD_01"],
    "pallet": ["MI_PaletteSmallA_01"],
    "box": ["MI_CardBoxA"],
}
ROOTS = ("/World/warehouse", "/World/loaded_pallet_01", "/World/loaded_pallet_02", "/World/loaded_pallet_03")


def _hsv(h, s, v):
    return colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))


def sample_colors(rng):
    """(rgb, roughness) per category. Hue h is in 0-1 (0 red, 0.08 orange/brown, 0.15 yellow, 0.33 green, 0.6 blue)."""
    out = {}
    # Floor: gray / beige / reddish-brown / blue-gray epoxy, excluding green (0.22-0.45)
    kind = rng.random()
    if kind < 0.45:
        out["floor"] = (_hsv(0, 0, rng.uniform(0.25, 0.75)), rng.uniform(0.4, 0.95))           # gray concrete
    elif kind < 0.7:
        out["floor"] = (_hsv(rng.uniform(0.07, 0.13), rng.uniform(0.15, 0.4), rng.uniform(0.45, 0.8)), rng.uniform(0.4, 0.9))
    elif kind < 0.85:
        out["floor"] = (_hsv(rng.uniform(0.98, 1.04), rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.55)), rng.uniform(0.3, 0.6))
    else:
        out["floor"] = (_hsv(rng.uniform(0.55, 0.62), rng.uniform(0.1, 0.3), rng.uniform(0.35, 0.65)), rng.uniform(0.3, 0.6))
    out["stripe"] = (_hsv(rng.choice([0.0, 0.13, 0.15]), rng.uniform(0, 0.9), rng.uniform(0.7, 0.95)), 0.5)
    # Walls: light neutral to beige to pale yellow/orange
    out["wall"] = (_hsv(rng.uniform(0.05, 0.16), rng.uniform(0.0, 0.45), rng.uniform(0.55, 0.95)), rng.uniform(0.6, 0.95))
    out["wall_detail"] = (_hsv(rng.uniform(0, 1), rng.uniform(0, 0.4), rng.uniform(0.2, 0.6)), 0.7)
    # Shelves and pillars: gray / orange / yellow / red / green, excluding blue (0.5-0.72)
    h = rng.choice([None, 0.07, 0.13, 0.0, 0.33])
    out["steel"] = ((_hsv(0, 0, rng.uniform(0.2, 0.7)) if h is None else _hsv(h + rng.uniform(-0.02, 0.02), rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.85))),
                    rng.uniform(0.3, 0.7))
    # Pallets: wood (light to dark) / black or gray plastic / red plastic, excluding blue
    k = rng.random()
    if k < 0.6:
        out["pallet"] = (_hsv(rng.uniform(0.06, 0.11), rng.uniform(0.3, 0.65), rng.uniform(0.3, 0.85)), rng.uniform(0.6, 0.95))
    elif k < 0.85:
        out["pallet"] = (_hsv(0, 0, rng.uniform(0.08, 0.5)), rng.uniform(0.4, 0.8))
    else:
        out["pallet"] = (_hsv(rng.uniform(-0.02, 0.02), rng.uniform(0.6, 0.85), rng.uniform(0.4, 0.7)), 0.5)
    # Boxes: cardboard brown (light to dark) / some whites / gray
    k = rng.random()
    if k < 0.7:
        out["box"] = (_hsv(rng.uniform(0.07, 0.11), rng.uniform(0.3, 0.6), rng.uniform(0.35, 0.85)), 0.85)
    elif k < 0.85:
        out["box"] = (_hsv(0, 0, rng.uniform(0.75, 0.92)), 0.8)
    else:
        out["box"] = (_hsv(0, 0, rng.uniform(0.3, 0.6)), 0.8)
    return out


class EnvRand:
    def __init__(self, stage, log=print, p_orig=0.25):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
        self.Gf, self.Sdf, self.UsdShade = Gf, Sdf, UsdShade
        self.stage, self.log, self.p_orig = stage, log, p_orig
        self.shaders, self.mats = {}, {}
        for cat in CATS:
            path = f"/World/_v3_looks/{cat}"
            m = UsdShade.Material.Define(stage, path)
            sh = UsdShade.Shader.Define(stage, path + "/s")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.shaders[cat], self.mats[cat] = sh, m
        name2cat = {n: c for c, ns in CATS.items() for n in ns}
        self.targets = []                                  # (prim, category)
        for root in ROOTS:
            rp = stage.GetPrimAtPath(root)
            if not rp:
                continue
            for p in Usd.PrimRange(rp):
                if p.IsInstanceable():
                    p.SetInstanceable(False)
            for p in Usd.PrimRange(rp):
                if not p.IsA(UsdGeom.Gprim):
                    continue
                b = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
                if b and b.GetPath().name in name2cat:
                    self.targets.append((p, name2cat[b.GetPath().name]))
        self.lights = []
        for p in stage.Traverse():
            if p.HasAPI(UsdLux.LightAPI) and str(p.GetPath()).startswith("/World/warehouse"):
                self.lights.append(p.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f))
        self.bound = False
        log(f"[env] randomization ready: target meshes {len(self.targets)}, lights {len(self.lights)}, original-appearance probability {p_orig}")

    def _bind(self, on):
        if on == self.bound:
            return
        for p, cat in self.targets:
            api = self.UsdShade.MaterialBindingAPI.Apply(p)
            if on:
                api.Bind(self.mats[cat], self.UsdShade.Tokens.strongerThanDescendants)
            else:
                api.UnbindDirectBinding()
        self.bound = on

    def episode(self, rng):
        """Call once per episode. Returns a description string for logging."""
        Gf = self.Gf
        if rng.random() < self.p_orig:
            self._bind(False)
            desc = "orig"
        else:
            cols = sample_colors(rng)
            for cat, (rgb, rough) in cols.items():
                sh = self.shaders[cat]
                sh.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(c) for c in rgb]))
                sh.GetInput("roughness").Set(float(rough))
            self._bind(True)
            desc = " ".join(f"{c}=({r:.2f},{g:.2f},{b:.2f})" for c, ((r, g, b), _) in cols.items() if c in ("floor", "pallet", "box", "steel"))
        # Light color temperature: warm to cool (brightness is randomized separately by --rand)
        t = rng.uniform(-1.0, 1.0)
        lc = (1.0, 1.0 - 0.15 * max(t, 0), 1.0 - 0.3 * max(t, 0)) if t > 0 else (1.0 + 0.0 * t, 1.0 + 0.05 * t, 1.0)
        lc = (lc[0] * (1 + 0.15 * min(t, 0)), lc[1], lc[2])
        for a in self.lights:
            a.Set(Gf.Vec3f(*[float(max(0.0, min(1.0, c))) for c in lc]))
        return f"{desc} light=({lc[0]:.2f},{lc[1]:.2f},{lc[2]:.2f})"
