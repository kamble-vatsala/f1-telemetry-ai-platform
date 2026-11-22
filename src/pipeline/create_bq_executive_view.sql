CREATE OR REPLACE VIEW `apex-racing-f1-bi.f1_executive_warehouse.vw_executive_scorecard` AS
WITH stint_summary AS (
  SELECT
    Year,
    RoundNumber,
    SessionType,
    Driver,
    COUNT(DISTINCT Stint) AS Total_Stints,
    ROUND(AVG(IF(IS_NAN(AvgFuelCorrectedPace_Sec), NULL, AvgFuelCorrectedPace_Sec)), 3) AS Avg_Fuel_Corrected_Pace_Sec,
    ROUND(AVG(IF(IS_NAN(TyreDegSlope_SecPerLap), NULL, TyreDegSlope_SecPerLap)), 4) AS Avg_Tyre_Deg_Slope,
    STRING_AGG(Compound, ' -> ' ORDER BY Stint) AS Tyre_Strategy_Sequence
  FROM `apex-racing-f1-bi.f1_executive_warehouse.fact_stints`
  GROUP BY Year, RoundNumber, SessionType, Driver
)
SELECT
  r.Year,
  r.RoundNumber,
  r.EventName,
  r.Country,
  r.SessionType,
  r.Driver,
  r.FullName,
  r.OfficialTeamName,
  r.TeamFranchise,
  r.GridPosition,
  r.FinishPosition,
  r.PositionsGained,
  r.Points,
  r.Status,
  CASE WHEN r.SessionType = 'R' AND r.FinishPosition <= 3 THEN 1 ELSE 0 END AS Podiums,
  CASE WHEN r.IsWetSession THEN 'Wet' ELSE 'Dry' END AS Track_Condition,
  r.SC_Count,
  r.VSC_Count,
  r.RedFlag_Count,
  NULLIF(
    LEAST(
      COALESCE(IF(IS_NAN(r.Q1_Sec), NULL, r.Q1_Sec), 9999.0),
      COALESCE(IF(IS_NAN(r.Q2_Sec), NULL, r.Q2_Sec), 9999.0),
      COALESCE(IF(IS_NAN(r.Q3_Sec), NULL, r.Q3_Sec), 9999.0)
    ),
    9999.0
  ) AS Best_Quali_Sec,
  s.Total_Stints,
  s.Avg_Fuel_Corrected_Pace_Sec,
  s.Avg_Tyre_Deg_Slope,
  s.Tyre_Strategy_Sequence,
  w.AvgTrackTemp,
  w.MaxTrackTemp,
  w.AvgWindSpeed
FROM `apex-racing-f1-bi.f1_executive_warehouse.dim_results` r
LEFT JOIN stint_summary s
  ON r.Year = s.Year
  AND r.RoundNumber = s.RoundNumber
  AND r.SessionType = s.SessionType
  AND r.Driver = s.Driver
LEFT JOIN `apex-racing-f1-bi.f1_executive_warehouse.dim_weather` w
  ON r.Year = w.Year
  AND r.RoundNumber = w.RoundNumber
  AND r.SessionType = w.SessionType;