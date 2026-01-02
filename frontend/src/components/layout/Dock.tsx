"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  BarChart3,
  Search,
  TrendingUp,
  Settings,
  Briefcase,
  History,
  ScanLine,
  Globe,
  LineChart,
  TestTube,
  Layers,
  Shield,
  Activity,
  Cog,
} from "lucide-react";

interface SubItem {
  name: string;
  href: string;
  icon: React.ReactNode;
}

interface DockCategory {
  name: string;
  icon: React.ReactNode;
  href?: string;
  items?: SubItem[];
}

const dockCategories: DockCategory[] = [
  {
    name: "Home",
    icon: <Home className="w-5 h-5" />,
    href: "/",
  },
  {
    name: "Portfolio",
    icon: <BarChart3 className="w-5 h-5" />,
    items: [
      { name: "Positions", href: "/positions", icon: <Briefcase className="w-4 h-4" /> },
      { name: "Trade History", href: "/history", icon: <History className="w-4 h-4" /> },
    ],
  },
  {
    name: "Markets",
    icon: <Search className="w-5 h-5" />,
    items: [
      { name: "Scanner", href: "/scanner", icon: <ScanLine className="w-4 h-4" /> },
      { name: "Market Overview", href: "/markets", icon: <Globe className="w-4 h-4" /> },
    ],
  },
  {
    name: "Insights",
    icon: <TrendingUp className="w-5 h-5" />,
    items: [
      { name: "Analytics", href: "/analytics", icon: <LineChart className="w-4 h-4" /> },
      { name: "Backtesting", href: "/backtest", icon: <TestTube className="w-4 h-4" /> },
      { name: "Strategies", href: "/strategies", icon: <Layers className="w-4 h-4" /> },
    ],
  },
  {
    name: "System",
    icon: <Settings className="w-5 h-5" />,
    items: [
      { name: "Risk Monitor", href: "/risk", icon: <Shield className="w-4 h-4" /> },
      { name: "Activity Feed", href: "/activity", icon: <Activity className="w-4 h-4" /> },
      { name: "Settings", href: "/settings", icon: <Cog className="w-4 h-4" /> },
    ],
  },
];

export function Dock() {
  const pathname = usePathname();
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      if (currentScrollY > lastScrollY && currentScrollY > 100) {
        setIsVisible(false);
      } else {
        setIsVisible(true);
      }

      setLastScrollY(currentScrollY);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [lastScrollY]);

  const isActive = (category: DockCategory) => {
    if (category.href) {
      return pathname === category.href;
    }
    return category.items?.some((item) => pathname === item.href);
  };

  const isItemActive = (href: string) => pathname === href;

  return (
    <div
      className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-50 transition-all duration-300 ${
        isVisible ? "translate-y-0 opacity-100" : "translate-y-full opacity-0"
      }`}
    >
      <nav
        className="glass px-3 py-2.5 flex items-center gap-1 animate-dock-in"
        style={{
          boxShadow: "0 0 40px rgba(0, 0, 0, 0.5), 0 0 60px rgba(16, 185, 129, 0.1)",
        }}
      >
        {dockCategories.map((category) => (
          <div
            key={category.name}
            className="relative"
            onMouseEnter={() => setHoveredCategory(category.name)}
            onMouseLeave={() => setHoveredCategory(null)}
          >
            {/* Main icon button */}
            {category.href ? (
              <Link
                href={category.href}
                className={`dock-item flex items-center justify-center w-11 h-11 rounded-xl transition-all duration-200 ${
                  isActive(category)
                    ? "text-emerald bg-emerald-glow active"
                    : "text-text-secondary hover:text-white hover:bg-surface-2"
                }`}
              >
                {category.icon}
              </Link>
            ) : (
              <button
                className={`dock-item flex items-center justify-center w-11 h-11 rounded-xl transition-all duration-200 ${
                  isActive(category)
                    ? "text-emerald bg-emerald-glow active"
                    : "text-text-secondary hover:text-white hover:bg-surface-2"
                }`}
              >
                {category.icon}
              </button>
            )}

            {/* Submenu */}
            {category.items && hoveredCategory === category.name && (
              <div
                className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 animate-slide-down"
                style={{ opacity: 0, animationFillMode: "forwards" }}
              >
                <div
                  className="glass px-2 py-2 min-w-[140px]"
                  style={{
                    boxShadow: "0 10px 40px rgba(0, 0, 0, 0.4)",
                  }}
                >
                  {/* Category label */}
                  <div className="px-2 py-1.5 text-xs font-medium text-text-muted uppercase tracking-wider mb-1">
                    {category.name}
                  </div>

                  {/* Submenu items */}
                  {category.items.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`flex items-center gap-2.5 px-2 py-2 rounded-lg transition-all duration-150 ${
                        isItemActive(item.href)
                          ? "text-emerald bg-emerald-glow"
                          : "text-text-secondary hover:text-white hover:bg-surface-2"
                      }`}
                    >
                      {item.icon}
                      <span className="text-sm font-medium whitespace-nowrap">
                        {item.name}
                      </span>
                    </Link>
                  ))}
                </div>

                {/* Arrow pointer */}
                <div
                  className="absolute left-1/2 -translate-x-1/2 -bottom-1.5 w-3 h-3 rotate-45"
                  style={{
                    background: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderTop: "none",
                    borderLeft: "none",
                  }}
                />

                {/* Invisible hover bridge - connects submenu to dock button */}
                <div className="absolute left-0 right-0 -bottom-3 h-4" />
              </div>
            )}

            {/* Hover label for items without submenu */}
            {category.href && hoveredCategory === category.name && (
              <div
                className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 animate-slide-down"
                style={{ opacity: 0, animationFillMode: "forwards" }}
              >
                <div
                  className="glass px-3 py-1.5"
                  style={{
                    boxShadow: "0 10px 40px rgba(0, 0, 0, 0.4)",
                  }}
                >
                  <span className="text-sm font-medium text-white whitespace-nowrap">
                    {category.name}
                  </span>
                </div>
                <div
                  className="absolute left-1/2 -translate-x-1/2 -bottom-1.5 w-3 h-3 rotate-45"
                  style={{
                    background: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderTop: "none",
                    borderLeft: "none",
                  }}
                />
              </div>
            )}
          </div>
        ))}
      </nav>
    </div>
  );
}
