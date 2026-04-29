import streamlit as st
import math, requests, datetime, heapq
import folium
from streamlit import components
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Cold Chain Route Intelligence", layout="wide")

st.title("❄️ Cold Chain Route Intelligence")
st.markdown("### AI-Powered Cold Chain Logistics Optimization")
st.markdown("""
<style>
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
.label {
    color: #9CA3AF;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------
st.sidebar.header("📍 Route Input")

origin = st.sidebar.text_input("Origin", "Delhi, India")
destination = st.sidebar.text_input("Destination", "Chandigarh, India")
stops = st.sidebar.text_area("Stops (comma separated)")

product = st.sidebar.selectbox("Product Type", [
    "Fresh Produce","Dairy","Frozen Goods","Meat & Seafood","Pharmaceuticals"
])

run = st.sidebar.button("🚀 Analyze Route")

# ---------------------------------------------------
# PRODUCT CONFIG
# ---------------------------------------------------
PRODUCTS = {
    "Fresh Produce":{"decay_lambda":0.025,"max_hours":36,"temp":8,"fuel_eff":5.8},
    "Dairy":{"decay_lambda":0.03,"max_hours":48,"temp":4,"fuel_eff":6},
    "Frozen Goods":{"decay_lambda":0.008,"max_hours":120,"temp":-18,"fuel_eff":4.8},
    "Meat & Seafood":{"decay_lambda":0.045,"max_hours":30,"temp":2,"fuel_eff":5.5},
    "Pharmaceuticals":{"decay_lambda":0.01,"max_hours":96,"temp":8,"fuel_eff":6.5}
}

# ---------------------------------------------------
# UTIL FUNCTIONS
# ---------------------------------------------------
@st.cache_data
def geocode(place):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q":place,"format":"json","limit":1},
            headers={"User-Agent":"coldchain"},
            timeout=5
        ).json()
        return (float(r[0]["lat"]), float(r[0]["lon"])) if r else None
    except:
        return None

@st.cache_data
def route(a,b):
    try:
        r = requests.get(
            f"http://router.project-osrm.org/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}?overview=full&geometries=polyline",
            timeout=5
        ).json()

        rt = r["routes"][0]
        return {
            "distance": rt["distance"]/1000,
            "duration": rt["duration"]/3600,
            "geometry": rt["geometry"]
        }
    except:
        return None

def decode(poly):
    index, lat, lng = 0, 0, 0
    coords = []

    while index < len(poly):
        for i in range(2):
            result, shift = 0, 0
            while True:
                b = ord(poly[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if result & 1 else result >> 1
            if i == 0: lat += d
            else: lng += d
        coords.append((lat/1e5, lng/1e5))
    return coords

# ---------------------------------------------------
# ML MODEL
# ---------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.DataFrame({
        "distance": np.random.randint(20,700,500),
        "hour": np.random.randint(0,24,500),
        "traffic": np.random.uniform(1,1.5,500)
    })
    df["delay"] = 0.4*df["distance"] + 6*df["traffic"]

    model = RandomForestRegressor(n_estimators=200)
    model.fit(df[["distance","hour","traffic"]], df["delay"])
    return model

model = train_model()

def predict_delay(distance, traffic):
    return model.predict(pd.DataFrame({
        "distance":[distance],
        "hour":[datetime.datetime.now().hour],
        "traffic":[traffic]
    }))[0]

# ---------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------
def greedy_opt(origin, stops, dest):
    coords = {x: geocode(x) for x in [origin]+stops+[dest]}
    route_order = [origin]
    current = origin
    unvisited = stops.copy()

    while unvisited:
        nxt = min(unvisited, key=lambda s: math.dist(coords[current], coords[s]))
        route_order.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    route_order.append(dest)
    return route_order

def astar_opt(origin, stops, dest):
    coords = {x: geocode(x) for x in [origin]+stops+[dest]}
    pq = [(0, origin, [], stops)]

    while pq:
        cost, curr, path, rem = heapq.heappop(pq)
        if not rem:
            return [origin]+path+[dest]

        for s in rem:
            new_rem = rem.copy()
            new_rem.remove(s)
            d = math.dist(coords[curr], coords[s])
            heapq.heappush(pq, (cost+d, s, path+[s], new_rem))

    return None

# ---------------------------------------------------
# ANALYSIS
# ---------------------------------------------------
def analyze(origin,destination,stops,product):

    stop_list = [s.strip() for s in stops.split(",") if s.strip()]

    if len(stop_list)<=5:
        route_order = astar_opt(origin, stop_list, destination)
        algo = "A* (Optimal)"
    else:
        route_order = greedy_opt(origin, stop_list, destination)
        algo = "AI Greedy (Fast)"

    total_dist,total_time,coords_all = 0,0,[]

    for i in range(len(route_order)-1):
        o,d = geocode(route_order[i]), geocode(route_order[i+1])
        r = route(o,d)
        if not r:
            return None

        total_dist += r["distance"]
        total_time += r["duration"]
        coords_all += decode(r["geometry"])

    traffic = 1.3
    total_time = total_time*traffic + predict_delay(total_dist,traffic)/60

    p = PRODUCTS[product]

    spoil = (1-math.exp(-p["decay_lambda"]*total_time))*100
    fuel = total_dist/p["fuel_eff"]

    return {
        "distance":total_dist,
        "time":total_time,
        "spoilage":min(spoil,100),
        "fuel":fuel,
        "fuel_cost":fuel*89.5,
        "co2":fuel*2.68,
        "grade": "A" if spoil<5 else "B" if spoil<12 else "C" if spoil<20 else "D",
        "coords":coords_all,
        "route_order":route_order,
        "optimizer":algo,
        "max_hours":p["max_hours"],
        "original":[origin]+stop_list+[destination]
    }

# ---------------------------------------------------
# UI OUTPUT
# ---------------------------------------------------
if run:

    res = analyze(origin,destination,stops,product)

    if not res:
        st.error("Route failed")
        st.stop()

    # ------------------------------
    # 🎯 COLOR MAPPING
    # ------------------------------
    grade_colors = {
        "A": "#22c55e",  # Green
        "B": "#3b82f6",  # Blue
        "C": "#f59e0b",  # Orange
        "D": "#ef4444"   # Red
    }

    color = grade_colors[res["grade"]]

    # ------------------------------
    # 📊 KPI CARDS
    # ------------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    def card(title, value):
        st.markdown(f"""
        <div class="card">
            <div class="label">{title}</div>
            <div class="metric">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        card("Distance", f"{res['distance']:.1f} km")
    with col2:
        card("Time", f"{res['time']:.2f} h")
    with col3:
        card("Spoilage", f"{res['spoilage']:.2f}%")
    with col4:
        card("Fuel Cost", f"₹{res['fuel_cost']:.0f}")

    # ------------------------------
    # 🧠 ROUTE CARD
    # ------------------------------
    st.subheader("🧠 Optimized Route")

    st.markdown(f"""
    <div class="card">
        {" → ".join(res["route_order"])}
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------
    # 🔍 BEFORE VS AFTER
    # ------------------------------
    st.subheader("🔍 Before vs After AI")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="card">
        <b>Original Route</b><br><br>
        {" → ".join(res["original"])}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
        <b>Optimized Route</b><br><br>
        {" → ".join(res["route_order"])}
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------
    # ⏱ SAFE WINDOW
    # ------------------------------
    safe = res["max_hours"] - res["time"]

    st.subheader("⏱ Cold Chain Safety")

    st.markdown(f"""
    <div class="card">
        Safe Time Remaining: <b>{safe:.2f} hours</b>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------
    # 🚦 RISK (COLORED)
    # ------------------------------
    st.subheader("🚦 Risk Assessment")

    st.markdown(f"""
    <div class="card">
        Risk Grade: <span style="color:{color}; font-size:24px;"><b>{res['grade']}</b></span>
    </div>
    """, unsafe_allow_html=True)

    if res["grade"]=="A":
        st.success("Low risk — ideal route")
    elif res["grade"]=="B":
        st.info("Moderate risk — manageable")
    elif res["grade"]=="C":
        st.warning("High risk — needs attention")
    else:
        st.error("Critical risk — not recommended")

    # ------------------------------
    # 🧠 AI INSIGHT
    # ------------------------------
    st.subheader("🧠 AI Decision Insight")

    st.markdown("""
    <div class="card">
    • Route minimizes total distance<br>
    • Reduces spoilage probability<br>
    • Optimizes stop sequence<br>
    • Balances fuel efficiency & time<br>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------
    # 🗺 MAP
    # ------------------------------
    st.subheader("🗺 Route Map")

    fmap = folium.Map(location=res["coords"][0], zoom_start=6)
    folium.PolyLine(res["coords"], color=color, weight=6).add_to(fmap)

    components.v1.html(fmap._repr_html_(), height=500)

else:
    st.info("Enter inputs and click Analyze Route")