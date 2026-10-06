These files contain evaluation data for the greenhouse-gas forecasting benchmark.

Monthly global mean mole fractions of four greenhouse gases, produced by the
NOAA Global Monitoring Laboratory (GML) cooperative air sampling network:

- `co2.csv`: carbon dioxide in parts per million (ppm), 1979-2026.
- `ch4.csv`: methane in parts per billion (ppb), 1983-2026.
- `n2o.csv`: nitrous oxide in parts per billion (ppb), 2001-2026.
- `sf6.csv`: sulfur hexafluoride in parts per trillion (ppt), 1997-2026.

Each file stores `decimal_year,monthly_mean` pairs with missing months removed.
The original records state: "These data are made freely available to the
public and the scientific community" (public-domain-style NOAA GML policy).
Cite the source, for example: Lan, X., et al. (2026), "Trends in globally-atmospheric
greenhouse gases", NOAA GML, gml.noaa.gov/ccgg/trends/.
Rows with missing values (-9.99 markers) were removed. All other values are unchanged.
