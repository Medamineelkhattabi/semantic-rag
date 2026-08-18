# Helios Aerospace — Quality Incident Log (FY2027 H1)

Document ID: DOC-INC-2027H1
Classification: Internal
Owner: Quality Assurance
Last revised: 2027-04-18

## Scope

This log records quality incidents raised against delivered hardware. Incidents
are recorded against the affected component identifier. Programme impact is
assessed separately by the PMO.

## INC-1042 — Calibration drift on delivered units

- Raised: 2027-02-09
- Severity: Major
- Status: Open — containment in place
- Affected component: C-17
- Investigating engineer: Tomás Herrera

Fourteen delivered C-17 units exhibited gyro bias drift exceeding the 0.008
deg/hr specification during incoming inspection, with the worst unit measured at
0.021 deg/hr. Incident INC-1042 affects component C-17. Root cause analysis
points to a change in the supplier's thermal soak profile introduced without
notification under the change-control clause. All fourteen units were
quarantined and a 100% incoming inspection regime was imposed.

The incident is directly relevant to the single-source exposure already recorded
in the enterprise register, because containment depends entirely on the same
manufacturer correcting its own process.

## INC-1077 — Bondline delamination

- Raised: 2027-01-22
- Severity: Critical
- Status: Open — corrective action in progress
- Affected component: C-58
- Investigating engineer: Tomás Herrera

Ultrasonic inspection of three C-58 panels revealed bondline void fractions of
up to 6.8% against a 2.0% acceptance limit. Incident INC-1077 affects component
C-58. The finding is consistent with the open action recorded against the
component's conditional qualification.

## INC-1103 — Ka-band packet loss under thermal load

- Raised: 2027-03-04
- Severity: Major
- Status: Closed — 2027-04-11
- Affected component: C-45
- Investigating engineer: Nadia Farouk

Two C-45 transceivers exhibited packet loss rising to 3.4% once case temperature
exceeded 58 °C during thermal-vacuum qualification. Incident INC-1103 affects
component C-45. The supplier identified a firmware timing defect in the adaptive
coding loop and issued a corrected build, which was verified and the incident
closed.

## INC-1150 — Overvoltage transient on power-up

- Raised: 2027-03-27
- Severity: Minor
- Status: Open — under investigation
- Affected component: C-22
- Investigating engineer: Nadia Farouk

A single C-22 unit produced a 41 V transient lasting 180 microseconds on cold
power-up, against a 32 V limit. Incident INC-1150 affects component C-22. No
downstream damage was observed and the transient has not been reproduced in 340
subsequent power cycles.

## INC-1168 — Servo backlash out of tolerance

- Raised: 2027-04-02
- Severity: Minor
- Status: Open — under investigation
- Affected component: C-77
- Investigating engineer: Nadia Farouk

Angular backlash of 0.047 degrees was measured on a C-77 servo drive against a
0.030-degree limit. Incident INC-1168 affects component C-77.

## Log notes

Incidents reference component identifiers only. The supplier responsible for a
given component is recorded in the Component Register, and the programmes
affected by a component are recorded in the Project Portfolio.
