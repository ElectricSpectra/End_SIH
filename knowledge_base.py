"""
POWERGRID Knowledge Base - Real Engineering Data & Industry Standards
Based on actual transmission line specifications and procurement patterns
"""

# --- BASELINE MATERIALS (220kV, 100km, Plain Terrain) ---
BASELINE_MATERIALS_220KV = {
    "Galvanized Lattice Steel (Towers)": {
        "qty_range": "590-640",
        "unit": "MT",
        "specs": "IS 2062/IS 802; angle steel + plates + connectors",
        "lead_time": "3-6 months",
        "category": "steel",
        "priority": "High",
        "notes": "Critical path item - Foundation for entire project"
    },
    "High-Tensile Bolts, Nuts, Washers": {
        "qty_range": "105000-130000",
        "unit": "Nos",
        "specs": "IS 12427/IS 1367, HDG",
        "lead_time": "1-3 months",
        "category": "hardware",
        "priority": "High",
        "notes": "For all member joints, including step bolts"
    },
    "Concrete for Foundations (M20/25)": {
        "qty_range": "4000-4200",
        "unit": "Cum",
        "specs": "IS 456; Avg. 14-16 m³ per tower",
        "lead_time": "1-3 months",
        "category": "concrete",
        "priority": "High",
        "notes": "Pad/raft/chimney type foundations"
    },
    "Reinforcement Steel (Fe500/415)": {
        "qty_range": "410-470",
        "unit": "MT",
        "specs": "IS 1786; Avg. 1.5% steel in concrete volume",
        "lead_time": "2-4 months",
        "category": "steel",
        "priority": "High",
        "notes": "Foundation reinforcement"
    },
    "ACSR Zebra Conductor": {
        "qty_range": "325-340",
        "unit": "MT",
        "specs": "IS 398; 1.625 kg/m, 3 phases × 2 circuits",
        "lead_time": "4-8 months",
        "category": "conductor",
        "priority": "High",
        "notes": "Main transmission conductor"
    },
    "Earth Wire (GSS/OPGW)": {
        "qty_range": "105-110",
        "unit": "km",
        "specs": "IS 2141/IEC; Per line length + spares",
        "lead_time": "3-6 months",
        "category": "conductor",
        "priority": "High",
        "notes": "Ground wire for lightning protection"
    },
    "Suspension/Tension Insulators (Disc Type)": {
        "qty_range": "22500-23500",
        "unit": "Nos",
        "specs": "IS 731; 14 discs × 6 strings × ~270 towers",
        "lead_time": "2-4 months",
        "category": "insulator",
        "priority": "High",
        "notes": "Porcelain/composite disc insulators"
    },
    "Insulator Hardware (Clamps, Sockets, Clevis)": {
        "qty_range": "1650",
        "unit": "Sets",
        "specs": "IS/IEC standards; All towers per string/block",
        "lead_time": "2-4 months",
        "category": "hardware",
        "priority": "Medium",
        "notes": "Complete hardware sets for insulators"
    },
    "Grounding Rods/Electrodes (Cu/GS)": {
        "qty_range": "270",
        "unit": "Nos",
        "specs": "IS 3043; 1 per tower typically",
        "lead_time": "1-2 months",
        "category": "hardware",
        "priority": "Medium",
        "notes": "Copper or galvanized steel rods"
    },
    "Earthing Cable/Leads (Cu/GS)": {
        "qty_range": "3250-4000",
        "unit": "M",
        "specs": "IS 3043; 10-15m per tower average",
        "lead_time": "1-2 months",
        "category": "hardware",
        "priority": "Medium",
        "notes": "Grounding system cables"
    },
    "Anti-Climbing Devices (Guards)": {
        "qty_range": "270",
        "unit": "Nos",
        "specs": "Galvanized steel spikes",
        "lead_time": "1-2 months",
        "category": "hardware",
        "priority": "Low",
        "notes": "Security and safety equipment"
    },
    "Danger Plates/Warning Markers": {
        "qty_range": "270",
        "unit": "Nos",
        "specs": "IS 2551; 1 per tower",
        "lead_time": "1 month",
        "category": "hardware",
        "priority": "Low",
        "notes": "Safety signage and identification"
    },
    "Number/Phase/Circuit Plates": {
        "qty_range": "270",
        "unit": "Nos",
        "specs": "ID/color coding per circuit/phase",
        "lead_time": "1 month",
        "category": "hardware",
        "priority": "Low",
        "notes": "Tower identification plates"
    },
    "Bird Guards / Aviation Markers": {
        "qty_range": "85-120",
        "unit": "Nos",
        "specs": "As per route/environmental need",
        "lead_time": "1 month",
        "category": "hardware",
        "priority": "Low",
        "notes": "Environmental compliance items"
    }
}

# --- VOLTAGE LEVEL MULTIPLIERS ---
VOLTAGE_MULTIPLIERS = {
    "132kV": {
        "description": "Lower voltage transmission",
        "steel_mult": 0.85,
        "concrete_mult": 0.94,
        "insulators_mult": 0.74,
        "tower_count_mult": 1.17,
        "notes": "More towers needed, lighter structures"
    },
    "220kV": {
        "description": "BASELINE - Standard transmission",
        "steel_mult": 1.0,
        "concrete_mult": 1.0,
        "insulators_mult": 1.0,
        "tower_count_mult": 1.0,
        "notes": "Reference configuration"
    },
    "400kV": {
        "description": "High voltage transmission",
        "steel_mult": 1.69,
        "concrete_mult": 1.57,
        "insulators_mult": 1.48,
        "tower_count_mult": 0.94,
        "notes": "+100% steel, +67% concrete vs 132kV"
    },
    "765kV": {
        "description": "Extra high voltage transmission",
        "steel_mult": 3.19,
        "concrete_mult": 2.83,
        "insulators_mult": 2.42,
        "tower_count_mult": 0.94,
        "notes": "+279% steel, +201% concrete vs 132kV"
    }
}

# --- TERRAIN TYPE MULTIPLIERS ---
TERRAIN_MULTIPLIERS = {
    "Plain": {
        "description": "BASELINE - Standard flat terrain",
        "steel_mult": 1.0,
        "concrete_mult": 1.0,
        "rebar_mult": 1.0,
        "engineering_reason": "Standard design parameters"
    },
    "Hilly": {
        "description": "Mountainous/valley terrain",
        "steel_mult": 1.47,
        "concrete_mult": 2.06,
        "rebar_mult": 2.44,
        "engineering_reason": "Rock anchoring, valley crossings, slope stabilization; taller towers for clearance"
    },
    "Coastal": {
        "description": "Marine environment",
        "steel_mult": 1.15,
        "concrete_mult": 1.30,
        "rebar_mult": 1.40,
        "engineering_reason": "Marine-grade materials, enhanced corrosion protection, saltwater resistance"
    },
    "Forest": {
        "description": "Dense vegetation areas",
        "steel_mult": 1.34,
        "concrete_mult": 1.22,
        "rebar_mult": 1.28,
        "engineering_reason": "Taller towers for tree clearance, environmental compliance, wildlife protection"
    },
    "Desert": {
        "description": "Arid sandy regions",
        "steel_mult": 0.96,
        "concrete_mult": 1.27,
        "rebar_mult": 1.37,
        "engineering_reason": "Longer spans possible but deeper sandy soil foundations required"
    }
}

# --- S-CURVE PROCUREMENT PATTERNS ---
S_CURVE_PATTERNS = {
    "steel": {
        "description": "Critical path - Front-loaded procurement",
        "phases": {
            "initiation": {
                "months": "1-3",
                "percentage": "65%",
                "breakdown": "Month 1: 18%, Month 2: 27%, Month 3: 37% (PEAK)"
            },
            "growth_maturity": {
                "months": "4-12",
                "percentage": "30%",
                "breakdown": "Steady decline from 15% to 5% monthly"
            },
            "completion": {
                "months": "13-16",
                "percentage": "5%",
                "breakdown": "Spares and closeout"
            }
        },
        "lead_time": "3-6 months",
        "order_timing": "Month -6 to -3 (before project start)",
        "procurement_strategy": "Bulk order immediately, lock prices with forward contracts",
        "risk_level": "HIGH - Price volatility, supply chain disruptions"
    },
    "conductor": {
        "description": "Staged delivery aligned with tower erection",
        "phases": {
            "manufacturing": {
                "months": "1-4",
                "percentage": "25%",
                "breakdown": "Month 1-2: 9% each, Month 3-4: 7% each"
            },
            "peak_installation": {
                "months": "5-8",
                "percentage": "55%",
                "breakdown": "Month 6: 29% (PEAK), Months 5,7,8: 8-10% each"
            },
            "final_sections": {
                "months": "9-12",
                "percentage": "15%",
                "breakdown": "Tapering installation"
            },
            "spares_testing": {
                "months": "13-16",
                "percentage": "5%",
                "breakdown": "Testing and spare parts"
            }
        },
        "lead_time": "4-8 months",
        "order_timing": "Month -8 to -4",
        "procurement_strategy": "Staggered POs with delivery milestones",
        "risk_level": "MEDIUM - Raw material (aluminum/copper) price sensitivity"
    },
    "insulator": {
        "description": "Just-in-time delivery for installation",
        "phases": {
            "initial_setup": {
                "months": "1-4",
                "percentage": "15%",
                "breakdown": "Month 1-2: 2% each, Month 3-4: 5-6% each"
            },
            "main_installation": {
                "months": "5-10",
                "percentage": "70%",
                "breakdown": "Month 7: 30% (PEAK), Other months: 8-10% each"
            },
            "testing_commissioning": {
                "months": "11-16",
                "percentage": "15%",
                "breakdown": "Final installation and testing"
            }
        },
        "lead_time": "2-4 months",
        "order_timing": "Month -4 to -2",
        "procurement_strategy": "Progressive delivery aligned with tower erection schedule",
        "risk_level": "MEDIUM - Quality testing delays, vendor capacity constraints"
    },
    "concrete": {
        "description": "Foundation-focused early procurement",
        "phases": {
            "foundation_blitz": {
                "months": "1-2",
                "percentage": "45%",
                "breakdown": "Month 1: 23%, Month 2: 22%"
            },
            "continued_work": {
                "months": "3-5",
                "percentage": "40%",
                "breakdown": "Gradual decline 15% to 10% monthly"
            },
            "repairs_additions": {
                "months": "6-12",
                "percentage": "15%",
                "breakdown": "Repairs and final touches"
            }
        },
        "lead_time": "1-3 months",
        "order_timing": "Month -3 to -1",
        "procurement_strategy": "Local sourcing preferred, just-in-time delivery",
        "risk_level": "MEDIUM - Weather dependency (monsoon), transportation logistics"
    },
    "hardware": {
        "description": "Continuous supply throughout project",
        "phases": {
            "early_phase": {
                "months": "1-6",
                "percentage": "40%",
                "breakdown": "Foundation and initial assembly hardware"
            },
            "mid_phase": {
                "months": "7-12",
                "percentage": "45%",
                "breakdown": "Main installation hardware"
            },
            "late_phase": {
                "months": "13-16",
                "percentage": "15%",
                "breakdown": "Final assembly and spares"
            }
        },
        "lead_time": "1-3 months",
        "order_timing": "Month -3 to -1",
        "procurement_strategy": "Phased procurement with vendor-managed inventory",
        "risk_level": "LOW - Standardized items, multiple suppliers"
    }
}

# --- EXAMPLE CALCULATIONS ---
EXAMPLE_CALCULATIONS = """
**EXAMPLE 1: 400kV, 200km, Hilly Terrain, 18-month project**

Material: Galvanized Steel
- Baseline (220kV, 100km, Plain): 615 MT (avg of 590-640)
- Scale Factor: 200km / 100km = 2.0x
- Voltage Multiplier (400kV): 1.69x
- Terrain Multiplier (Hilly): 1.47x
- **Final Calculation: 615 × 2.0 × 1.69 × 1.47 = 3,053 MT**

Monthly Procurement (Steel S-curve):
- Mo1: 550 MT (18%), Mo2: 824 MT (27%), Mo3: 1,130 MT (37% PEAK)
- Mo4-12: 916 MT total (30%), Mo13-18: 153 MT (5%)

Material: ACSR Conductor  
- Baseline: 332.5 MT
- Scale Factor: 2.0x
- Voltage Adjustment: 1.1x (minor for conductors)
- Terrain: 1.0x (not affected)
- **Final: 332.5 × 2.0 × 1.1 × 1.0 = 732 MT**

Monthly Procurement (Conductor S-curve):
- Mo1-4: 183 MT (25%), Mo5-8: 403 MT (55%, Peak Mo6: 212 MT)
- Mo9-12: 110 MT (15%), Mo13-18: 37 MT (5%)

**EXAMPLE 2: 220kV, 100km, Plain Terrain, 12-month project (Baseline)**

Material: Concrete for Foundations
- Baseline: 4,100 m³ (avg)
- Scale Factor: 1.0x (100km baseline)
- Voltage: 1.0x (220kV baseline)
- Terrain: 1.0x (Plain baseline)
- **Final: 4,100 m³ (no change)**

Monthly Procurement (Concrete S-curve):
- Mo1: 943 m³ (23%), Mo2: 902 m³ (22%)
- Mo3-5: 1,640 m³ (40%), Mo6-12: 615 m³ (15%)

Order Timeline:
- Steel: Order Month -6 (Lead time: 6 months)
- Conductor: Order Month -6 (Lead time: 6 months)  
- Insulators: Order Month -3 (Lead time: 3 months)
- Concrete: Order Month -2 (Local sourcing, 2-month lead time)
"""

# --- PROCUREMENT BEST PRACTICES ---
PROCUREMENT_BEST_PRACTICES = """
**POWERGRID PROCUREMENT GUIDELINES:**

1. **Lead Time Management:**
   - Steel Structures: 3-6 months → Order immediately upon project approval
   - Conductors (ACSR): 4-8 months → Stagger orders for delivery alignment
   - Insulators: 2-4 months → Order after tower design finalization
   - Concrete/Foundation: 1-3 months → Local sourcing, just-in-time

2. **Bulk Procurement Discounts:**
   - Steel: >500 MT → 5-8% discount
   - Conductors: >200 MT → 6-10% discount
   - Insulators: >20,000 units → 4-7% discount

3. **Multi-Vendor Strategy:**
   - Critical items (Steel, Conductors): Minimum 2 vendors (60-40 split)
   - Standard items (Hardware): 3+ vendors for redundancy
   - Specialized equipment (Transformers): Pre-qualified vendor list

4. **Price Volatility Hedging:**
   - Steel: Forward contracts for 6-12 months (coal-linked pricing)
   - Copper/Aluminum: LME-linked contracts with caps
   - Cement/Concrete: Annual rate contracts with escalation clauses

5. **Quality Assurance:**
   - All materials: IS/IEC standards mandatory
   - Steel: Mill test certificates + third-party testing
   - Conductors: Type testing + routine testing per IS 398
   - Insulators: Electrical and mechanical testing per IS 731

6. **Regional Considerations:**
   - Hilly terrain: Helicopter/ropeway logistics factored in cost
   - Coastal: Marine-grade materials (15-30% premium)
   - Remote areas: Buffer stock 20-30% higher
   - Monsoon zones: Accelerated pre-monsoon procurement
"""