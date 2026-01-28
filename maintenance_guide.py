# 🔧 MAINTENANCE RECOMMENDATIONS DATABASE
# Solar Panel Fault Classification & Maintenance Guide

MAINTENANCE_GUIDE = {
    "Clean": {
        "status": "✅ NORMAL",
        "color": "#2ecc71",  # Green
        "urgency": "Low",
        "description": "Panel is in perfect condition, no damage detected.",
        "actions": [
            "✓ Continue routine monitoring",
            "✓ Scheduled cleaning: every 3 months",
            "✓ Visual inspection: monthly",
        ],
        "maintenance_schedule": "Preventive (3 months)",
        "estimated_efficiency_loss": "0%",
        "recommended_actions": [
            {
                "action": "Preventive Cleaning",
                "frequency": "Every 3 months",
                "cost": "💰 Low (~$50-100/panel)",
                "details": "Clean with demineralized water + soft brush"
            },
            {
                "action": "Regular Inspection",
                "frequency": "Monthly",
                "cost": "💰 Free (visual check)",
                "details": "Check physical condition, mounting system, cables"
            }
        ]
    },
    
    "Dusty": {
        "status": "⚠️  WARNING - DUST",
        "color": "#f39c12",  # Orange
        "urgency": "Medium",
        "description": "Panel covered with dust/sand that reduces energy efficiency.",
        "actions": [
            "⚠ Urgent cleaning within 1-2 weeks",
            "⚠ Efficiency reduced ~10-25%",
            "⚠ Prioritize if rain occurs",
        ],
        "maintenance_schedule": "Immediate (1-2 weeks)",
        "estimated_efficiency_loss": "10-25%",
        "recommended_actions": [
            {
                "action": "Immediate Cleaning",
                "frequency": "Within 1-2 weeks",
                "cost": "💰 Low (~$50-100/panel)",
                "details": [
                    "• Use low-pressure water (max 80 bar)",
                    "• Soft brush or microfiber cloth",
                    "• Avoid cleaning during intense sunlight (thermal damage)",
                    "• Use demineralized water to avoid mineral spots"
                ]
            },
            {
                "action": "Root Cause Analysis",
                "frequency": "During cleaning",
                "cost": "💰 Free",
                "details": [
                    "• Check panel location (near road = more frequent dirt)",
                    "• Monitor seasonal dust patterns",
                    "• Consider installing screen/cover if in dusty area"
                ]
            },
            {
                "action": "Maintenance Plan",
                "frequency": "Adjusted",
                "cost": "💰 Medium",
                "details": "If very dusty area: cleaning every 1-2 months"
            }
        ]
    },
    
    "Bird-drop": {
        "status": "⚠️  WARNING - BIRD DROPPINGS",
        "color": "#e67e22",  # Dark Orange
        "urgency": "High",
        "description": "Panel contaminated with bird droppings which are acidic & damaging.",
        "actions": [
            "🔴 Urgent cleaning within 3-5 days",
            "🔴 Risk of surface damage (corrosive)",
            "🔴 Efficiency reduced ~20-35%",
        ],
        "maintenance_schedule": "Urgent (3-5 days)",
        "estimated_efficiency_loss": "20-35%",
        "recommended_actions": [
            {
                "action": "Emergency Cleaning",
                "frequency": "Within 3-5 days",
                "cost": "💰 Medium (~$100-200/panel)",
                "details": [
                    "• DO NOT leave > 7 days (permanent corrosion)",
                    "• Use pH-neutral cleaner or mild detergent",
                    "• Soak stubborn areas with wet cloth for 10 minutes",
                    "• Be careful with water pressure (can damage seal)",
                ]
            },
            {
                "action": "Damage Inspection",
                "frequency": "Before & after cleaning",
                "cost": "💰 Free (visual)",
                "details": [
                    "• Check for etching/pitting on glass",
                    "• Check junction box & connectors for corrosion",
                    "• Document with photos if damage exists"
                ]
            },
            {
                "action": "Long-term Prevention",
                "frequency": "One-time or ongoing",
                "cost": "💰 Medium-High ($200-500/panel)",
                "details": [
                    "• Install bird spikes/netting around panel area",
                    "• Keep area clear of trees/bird perching spots",
                    "• Install motion sensor or sound deterrent",
                    "• Frequent monitoring (weekly if in high-risk area)"
                ]
            }
        ]
    },
    
    "Snow-Covered": {
        "status": "❄️  SNOW COVERED",
        "color": "#3498db",  # Blue
        "urgency": "High",
        "description": "Panel covered with snow, unable to produce energy.",
        "actions": [
            "🔴 Immediate cleaning (if safe)",
            "🔴 Energy output: 0% (cannot generate)",
            "🔴 Wait for snow to melt or clean manually",
        ],
        "maintenance_schedule": "Immediate or wait for weather (3-7 days)",
        "estimated_efficiency_loss": "100%",
        "recommended_actions": [
            {
                "action": "Manual Cleaning (If Safe)",
                "frequency": "When heavy snow present",
                "cost": "💰 Medium (technician service)",
                "details": [
                    "⚠️  SAFETY FIRST: Use harness & working at height protocols",
                    "• Don't climb when structure is wet/slippery",
                    "• Use soft brush or rubber blade (not metal)",
                    "• Avoid scratching tempered glass surface",
                    "• Clean when temperature above 0°C (easier removal)"
                ]
            },
            {
                "action": "Alternative: Wait Naturally",
                "frequency": "Seasonal",
                "cost": "💰 Free (but lost revenue)",
                "details": [
                    "• Let snow melt naturally (safer option)",
                    "• Monitor weather forecast for clearing predictions",
                    "• Document downtime for revenue loss estimation",
                ]
            },
            {
                "action": "Prevention Infrastructure",
                "frequency": "One-time investment",
                "cost": "💰 High ($500-2000/panel)",
                "details": [
                    "• Install heated panels or self-cleaning coating",
                    "• Optimize tilt angle for self-shedding (30-35° ideal)",
                    "• Hydrophobic/oleophobic coating for easier melting"
                ]
            }
        ]
    },
    
    "Electrical-damage": {
        "status": "🔴 CRITICAL - ELECTRICAL DAMAGE",
        "color": "#e74c3c",  # Red
        "urgency": "CRITICAL",
        "description": "Panel experiencing electrical damage (possibly from surge/arc flash). Safety risk!",
        "actions": [
            "🔴 ISOLATE PANEL IMMEDIATELY - DO NOT USE",
            "🔴 Contact qualified technician ASAP",
            "🔴 Potential fire hazard & personal injury",
        ],
        "maintenance_schedule": "EMERGENCY (24 hours)",
        "estimated_efficiency_loss": "100% (or risk of total system damage)",
        "recommended_actions": [
            {
                "action": "Emergency Response",
                "frequency": "Immediately",
                "cost": "💰 CRITICAL",
                "details": [
                    "🚨 DISCONNECT panel from inverter (isolate electrical)",
                    "🚨 DO NOT touch panel or wiring",
                    "🚨 Call licensed electrician/solar technician",
                    "🚨 Document visual damage with photos (no contact)"
                ]
            },
            {
                "action": "Diagnostic Check",
                "frequency": "By qualified technician",
                "cost": "💰 Medium ($200-400)",
                "details": [
                    "• Thermal imaging to detect internal damage",
                    "• Electrical testing (IV curve, insulation resistance)",
                    "• Bypass diode check (broken = cannot repair, need replace)",
                    "• Junction box integrity assessment"
                ]
            },
            {
                "action": "Repair or Replace",
                "frequency": "Based on diagnosis",
                "cost": "💰 High ($300-1000+)",
                "details": [
                    "• Minor damage (junction box, connector): Can be repaired",
                    "• Major damage (internal circuit): Replace panel",
                    "• Claim insurance if coverage exists for electrical damage"
                ]
            }
        ]
    },
    
    "Physical-Damage": {
        "status": "🔴 CRITICAL - PHYSICAL DAMAGE",
        "color": "#c0392b",  # Dark Red
        "urgency": "CRITICAL",
        "description": "Panel broken/physically damaged (impact, weather, manufacturing defect).",
        "actions": [
            "🔴 ISOLATE PANEL - WATER INGRESS RISK",
            "🔴 Potential leakage & short circuit",
            "🔴 Immediate replacement required",
        ],
        "maintenance_schedule": "EMERGENCY (24-48 hours)",
        "estimated_efficiency_loss": "100% (risk of more damage if not isolated)",
        "recommended_actions": [
            {
                "action": "Immediate Isolation",
                "frequency": "Immediately",
                "cost": "💰 Minimal",
                "details": [
                    "⚠️  Isolate panel from system (turn off DC breaker)",
                    "⚠️  Cover broken area with protective tape/tarp",
                    "⚠️  Prevent water penetration into junction box",
                    "⚠️  Safety: Broken glass can cause cuts/injury"
                ]
            },
            {
                "action": "Damage Assessment",
                "frequency": "Within 24 hours",
                "cost": "💰 Free (visual inspection)",
                "details": [
                    "• Document damage with detailed photos",
                    "• Determine repair vs replace decision",
                    "• Check if under warranty (manufacturing/impact)",
                    "• Assess adjacent panels for damage propagation"
                ]
            },
            {
                "action": "Replacement",
                "frequency": "ASAP",
                "cost": "💰 Very High ($400-800 + labor)",
                "details": [
                    "• Broken panels cannot be repaired → must be replaced",
                    "• Order replacement panel (lead time 2-4 weeks)",
                    "• Hire licensed technician for replacement",
                    "• Verify new panel matches original specs (voltage, power)"
                ]
            },
            {
                "action": "Prevention Strategies",
                "frequency": "Going forward",
                "cost": "💰 Medium ($100-300/panel setup)",
                "details": [
                    "• Install protective barriers if in high-impact area",
                    "• Improve racking/mounting for weather resistance",
                    "• Tree trimming around array",
                    "• Hail/weather insurance for industrial deployments"
                ]
            }
        ]
    }
}

def get_maintenance_info(class_name):
    """Get maintenance recommendation untuk fault class"""
    return MAINTENANCE_GUIDE.get(class_name, {})

def get_all_classes():
    """Get all fault classes"""
    return list(MAINTENANCE_GUIDE.keys())

def get_urgency_priority(class_name):
    """Get urgency level (1-5, where 5 = critical)"""
    urgency_map = {
        "Clean": 1,
        "Dusty": 2,
        "Bird-drop": 3,
        "Snow-Covered": 3,
        "Electrical-damage": 5,
        "Physical-Damage": 5
    }
    return urgency_map.get(class_name, 0)

def get_priority_color(class_name):
    """Get color untuk UI based on fault type"""
    color_map = {
        "Clean": "#2ecc71",  # Green
        "Dusty": "#f39c12",  # Orange
        "Bird-drop": "#e67e22",  # Dark Orange
        "Snow-Covered": "#3498db",  # Blue
        "Electrical-damage": "#e74c3c",  # Red
        "Physical-Damage": "#c0392b"  # Dark Red
    }
    return color_map.get(class_name, "#95a5a6")
