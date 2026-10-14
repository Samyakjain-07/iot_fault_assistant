# Induction Motor & Energy Meter Fault Reference Guide

Device class: three-phase induction motor with an inline energy meter
(voltage, current, power, power factor, temperature). This guide is the
seed knowledge base for the diagnosis assistant — it is intentionally
written in prose, section by section, so it chunks and embeds cleanly.

## Device Fact Sheet

Rated voltage: 400V (three-phase). Rated current: 15A. Rated power: 7.5kW.
Nominal power factor: 0.90–0.95 under normal load. Normal operating
temperature range: 40–65°C at the motor housing. Readings are sampled every
few seconds from the inline meter.

## High Voltage

A sustained voltage reading meaningfully above 400V, especially paired with
a normal or slightly reduced current, usually points to an upstream supply
problem rather than the motor itself: an unstable grid feed, a
transformer tap set incorrectly, or a faulty voltage regulator on the
distribution panel feeding this circuit. Prolonged overvoltage stresses the
winding insulation and can shorten motor life even if the motor keeps
running normally in the short term. Recommended first step: check the
incoming supply voltage at the panel, not at the motor terminals, to confirm
whether the problem is upstream or local. If the supply is stable at the
panel but the motor terminals still read high, suspect a loose or
miswired connection between the panel and the motor.

## Low Voltage

A sustained drop below roughly 380V, especially during start-up or when
other loads come online on the same circuit, points to undersized supply
wiring, a loose connection somewhere in the feed, or the circuit being
overloaded by other equipment sharing the same supply. Low voltage makes
the motor draw more current to deliver the same power, which shows up as
elevated current and reduced power factor at the same time — if you see all
three together, voltage sag is the likely root cause, not a motor fault.
Recommended first step: check for other heavy loads starting on the same
circuit and inspect connection points for corrosion or looseness.

## High Current

Current meaningfully above the 15A rating, with voltage and temperature
both in their normal ranges, usually indicates a mechanical problem: the
motor is working harder than it should to turn its load. Common causes are
a partially seized bearing, debris or misalignment in the driven
equipment, or the load itself increasing (for example a conveyor jamming
partway). If high current appears together with a rising temperature trend
over several readings rather than a single spike, treat it as a developing
mechanical fault rather than a transient load event. Recommended first
step: check for unusual noise or vibration at the motor and inspect the
driven equipment for obstruction before assuming an electrical fault.

## High Temperature

A motor housing temperature trending above roughly 65°C, especially when
current is also elevated, points to one of: restricted ventilation around
the motor (blocked air intake, nearby heat sources, enclosure with no
airflow), a bearing starting to fail (friction generates heat even without
a current spike), or sustained overload. A temperature rise with current
staying normal is more likely a ventilation or bearing problem than an
electrical one. A temperature rise that tracks current closely is more
likely a load problem. Recommended first step: check airflow around the
motor first, since it is the fastest thing to rule out, then check bearing
condition if temperature keeps climbing with normal current.

## Low Power Factor / High Reactive Load

A power factor dropping meaningfully below the 0.90–0.95 normal range,
without a corresponding drop in voltage, usually means the motor is
running lightly loaded relative to its rating, or there's a developing
winding fault increasing reactive current draw. A lightly loaded motor
running at low power factor is often not an emergency — it's worth noting
for efficiency reasons but rarely indicates imminent failure. A low power
factor that appears suddenly, especially paired with unusual current
fluctuation, is more concerning and worth an electrical inspection.
Recommended first step: compare the load the motor is actually driving
against its rated capacity before assuming a fault.

## When Multiple Signals Appear Together

Readings rarely go wrong in isolation. As a rule of thumb for prioritizing
which cause to investigate first: voltage problems tend to show up with
current and power factor moving together; mechanical problems tend to show
current and temperature rising together while voltage and power factor
stay normal; and a developing electrical winding fault tends to show
temperature and power factor moving together while current stays close to
normal. Use that pattern as a starting hypothesis, not a certainty — always
state which readings led to the hypothesis so the technician can verify it
on-site.
