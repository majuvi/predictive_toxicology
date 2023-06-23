# Author: Pim Wassenaar

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Section 1: load packages ----
library(readr)
library(dplyr)
library(webchem)
library(readxl)
library(rcdk)



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Section 2: read data ----
tests <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/tests.txt", 
                    delim = "|", escape_double = FALSE, trim_ws = TRUE)

results <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/results.txt", 
                    delim = "|", escape_double = FALSE, trim_ws = TRUE)

#doses <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/doses.txt", 
#                    delim = "|", escape_double = FALSE, trim_ws = TRUE)

chemicals <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/validation/chemicals.txt", 
                      delim = "|", escape_double = FALSE, trim_ws = TRUE)

species <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/validation/species.txt", 
                      delim = "|", escape_double = FALSE, trim_ws = TRUE)

species_synonyms <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/validation/species_synonyms.txt", 
                               delim = "|", escape_double = FALSE, trim_ws = TRUE)

references <- read_delim("MIGRATION_data_curation/SECTION 2 - input/ecotox_ascii_09_15_2022/validation/references.txt", 
                         delim = "|", escape_double = FALSE, trim_ws = TRUE)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Section 3: transform and filter data ----

# @@@@@@@@@@@@@@
## Section 3.1: combine data files ----
ECOTOX <- merge(tests, results, by = "test_id", all.x = TRUE) #Combine tests information with results information
ECOTOX <- merge(ECOTOX, chemicals, by.x = "test_cas", by.y = "cas_number", all.x = TRUE) #Combine information with chemical information
ECOTOX <- merge(ECOTOX, species, by = "species_number", all.x = TRUE) #Combine information with species information
ECOTOX <- merge(ECOTOX, references, by = "reference_number", all.x = TRUE) #Combine information with reference/source information


# @@@@@@@@@@@@@@
## Section 3.2: extract relevant information ----
ECOTOX_MIGRATION <- ECOTOX[,c("test_id", "result_id", 
                            "test_cas", "chemical_name", "dtxsid","test_grade", "test_purity_mean_op", "test_purity_mean", "test_purity_comments", "chem_analysis_method", 
                            "ion1", "ion2", "ion3", 
                            "num_doses_mean_op", "num_doses_mean", "num_doses_comments", "test_location", "media_type", "media_type_comments","organism_habitat","exposure_type", 
                            "exposure_type_comments", "study_type", "test_method", "test_type", "application_type",
                            "common_name", "latin_name", "kingdom", "phylum_division", "subphylum_div", "superclass","class","tax_order", "family","genus","ncbi_taxid",
                            "obs_duration_mean_op", "obs_duration_mean", "obs_duration_unit", "obs_duration_comments",
                            "endpoint", "endpoint_comments", "endpoint_assigned", "effect", "effect_comments", 
                            "conc1_type", "conc1_mean_op", "conc1_mean", "conc1_unit", "conc1_comments",
                            "reference_db", "author", "title", "source", "publication_year")] 


# @@@@@@@@@@@@@@
## Section 3.3: filter ECOTOX dataset ----

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, endpoint == 'LC50') #Filter LC50

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, effect == 'MOR' | effect == '~MOR'| effect == '~MOR/' | effect == 'MOR/') #Filter effect is MORTALITY

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, organism_habitat == "Water") # filter organism habitat

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, is.na(conc1_mean_op)) # filter conc1_mean_op (exclude values '<' or '>' and '~')

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, test_location == "LAB" | test_location == "NR")

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, exposure_type != "FD" & exposure_type != "IJ" & exposure_type != "IP" & exposure_type != "IV" & exposure_type != "IVT" & exposure_type != "YK") 
# Exclude FD = food; IJ = injection, unspecified; IP = intraperitoneal; IV = Intravenous; IVT = In Vitro; YK = yolk inject. (Included are AQUA and ENV considered exposures and: MU = multiple routes; TP = topical, general = application route not specified)


ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, conc1_mean != "NR") # Exclude studies with no reported concentration.
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, !is.na(as.numeric(ECOTOX_MIGRATION$conc1_mean))) # Exclude all studies with concentrations including '*' or '>', etc. 
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, conc1_mean != 0) # Exclude studies with not exposure (conc = 0)

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, conc1_unit == "AI mg/L" | conc1_unit == "AI ppm" | conc1_unit == "AI ug/ml" | conc1_unit == "mg/L" | conc1_unit == "ppm" | conc1_unit == "ug/ml" |
                             conc1_unit == "AI ng/mL" | conc1_unit == "AI ppb" | conc1_unit == "AI ug/L" | conc1_unit == "ng/ml" | conc1_unit == "pg/ul" | conc1_unit == "ppb" | conc1_unit == "ug/L" |
                             conc1_unit == "AI ng/L" | conc1_unit == "ng/L" | conc1_unit == "pg/ml" |  
                             conc1_unit == "pg/L" |  conc1_unit == "ppt" |
                             conc1_unit == "g/L" | conc1_unit == "mg/ml")
# Included:  AI mg/L; AI ng/L; AI ng/mL; AI ppb; AI ppm; AI ug/L; AI ug/ml; g/L; mg/L; mg/ml; ng/L; ng/ml; pg/L; pg/ml; pg/ul; ppb; ppm; ppt; ug/L; ug/ml
# Examples excluded: ae mg/L; ae ug/L (acid equivalents milligrams per liter); M; mg/L 10 mi;mg/L/h; mM; nM; mmol/L; ng/0.5 ml; NR; ug/100 ml
# table(ECOTOX_MIGRATION$conc1_unit) 


ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, obs_duration_mean != "NR") # Exclude studies with no reported duration.
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, obs_duration_mean != 0) # Exclude studies with no exposure (duration = 0)
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, is.na(obs_duration_mean_op)) # filter obs_duration_mean_op (exclude values '<' or '>' and '~')

ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, obs_duration_unit == "d" | obs_duration_unit == "dpf" | obs_duration_unit == "h" | obs_duration_unit == "hpf" | 
                             obs_duration_unit == "mi" | obs_duration_unit == "mo" | obs_duration_unit == "wk")
#Include: d, dpf,  h, hpf,  mi (minutes), mo (months), wk (weeks)
#Exclude: s (seconds)?, brd, dph: days post hatch?, emergence, fry, generation, gosner stage, hph?,ht (until hatch), harvest, larva to adult, larva to pupa, larva to subadult, maturity, nieuwkoop faber stage, NR, post molt, stage, zoeae-megalop 
#table(ECOTOX_MIGRATION$obs_duration_unit)



# @@@@@@@@@@@@@@
## Section 3.4: standardize ECOTOX dataset ----

### Duration:
# standardize duration (d) 
table(ECOTOX_MIGRATION$obs_duration_unit)
ECOTOX_MIGRATION$obs_duration_mean_standardized_d <- ifelse(ECOTOX_MIGRATION$obs_duration_unit == 'h', as.numeric(ECOTOX_MIGRATION$obs_duration_mean)/24,
                                                            ifelse(ECOTOX_MIGRATION$obs_duration_unit == 'hpf', as.numeric(ECOTOX_MIGRATION$obs_duration_mean)/24,
                                                                   ifelse(ECOTOX_MIGRATION$obs_duration_unit == 'mi', as.numeric(ECOTOX_MIGRATION$obs_duration_mean)/60/24, 
                                                                          ifelse(ECOTOX_MIGRATION$obs_duration_unit == 'wk', as.numeric(ECOTOX_MIGRATION$obs_duration_mean)*7, 
                                                                                 ifelse(ECOTOX_MIGRATION$obs_duration_unit == 'mo', as.numeric(ECOTOX_MIGRATION$obs_duration_mean)*30,ECOTOX_MIGRATION$obs_duration_mean)))))
# filter duration less or equal to 7 days
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, as.numeric(obs_duration_mean_standardized_d) <= 7)

# filter duration more or equal to 1 hour (= 0.04166667 days)
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, round(as.numeric(ECOTOX_MIGRATION$obs_duration_mean_standardized_d), digits=5) >= round((1/24), digits=5))
#summary(as.numeric(ECOTOX_MIGRATION$obs_duration_mean_standardized_d))



### Concentration
# standardize concentration (mg/L) 
table(ECOTOX_MIGRATION$conc1_unit)
ECOTOX_MIGRATION$conc1_mean_standardized_mgL <- ifelse(ECOTOX_MIGRATION$conc1_unit == 'g/L', as.numeric(ECOTOX_MIGRATION$conc1_mean)*1000,
                                                       ifelse(ECOTOX_MIGRATION$conc1_unit == 'mg/ml', as.numeric(ECOTOX_MIGRATION$conc1_mean)*1000,
                                                ifelse(ECOTOX_MIGRATION$conc1_unit == 'ug/L', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000,
                                                       ifelse(ECOTOX_MIGRATION$conc1_unit == 'AI ug/L', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000,
                                                              ifelse(ECOTOX_MIGRATION$conc1_unit == 'ppb', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000, 
                                                                     ifelse(ECOTOX_MIGRATION$conc1_unit == 'AI ppb', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000,
                                                                            ifelse(ECOTOX_MIGRATION$conc1_unit == 'ng/ml', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000, 
                                                                                   ifelse(ECOTOX_MIGRATION$conc1_unit == 'AI ng/mL', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000,
                                                                                          ifelse(ECOTOX_MIGRATION$conc1_unit == 'pg/ul', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000,
                                                ifelse(ECOTOX_MIGRATION$conc1_unit == 'ng/L', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000000, 
                                                       ifelse(ECOTOX_MIGRATION$conc1_unit == 'AI ng/L', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000000, 
                                                              ifelse(ECOTOX_MIGRATION$conc1_unit == 'pg/ml', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000000, 
                                                ifelse(ECOTOX_MIGRATION$conc1_unit == 'pg/L', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000000000, 
                                                       ifelse(ECOTOX_MIGRATION$conc1_unit == 'ppt', as.numeric(ECOTOX_MIGRATION$conc1_mean)/1000000000, ECOTOX_MIGRATION$conc1_mean))))))))))))))


#filter with a max of 10 g/L (= 10000 mg/L)
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, as.numeric(conc1_mean_standardized_mgL) <= 10000)

#filter with a min of 1 pg/L (= 1e-9 mg/L) # or 1e-6 mg/L (= 1 ng/L)
ECOTOX_MIGRATION <- filter(ECOTOX_MIGRATION, as.numeric(conc1_mean_standardized_mgL) >= 1e-9) 
#summary(as.numeric(ECOTOX_MIGRATION$conc1_mean_standardized_mgL))




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Section 4: Export CHEMICAL dataset ----
length(unique(ECOTOX_MIGRATION$test_cas))
length(unique(ECOTOX_MIGRATION$common_name))
ECOTOX_MIGRATION$CAS <- as.cas(ECOTOX_MIGRATION$test_cas)

CHEMICALS_MIGRATION <- distinct(ECOTOX_MIGRATION[,c(3:5,59)], test_cas, .keep_all = TRUE)

## Export
write_csv(CHEMICALS_MIGRATION, "MIGRATION_data_curation/SECTION 4 - add and standardize SMILES/CHEMICALS_MIGRATION_2022-11-14.csv")

## Note: data are exported and imported (Section 5) to add SMILES via QSARr workflow to ensure standardized and neutralized SMILES. 


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Section 5: Import CHEMICAL dataset with QSARr SMILES ----

## SMILES derived from: https://epa.figshare.com/articles/dataset/Chemistry_Dashboard_Data_DSSTox_QSAR_Ready_File/6253679
## QSAR ready SMILES generated using: https://hub.knime.com/kmansouri/spaces/Public/latest/QSAR-ready_2.5.8~5TRvnGfMJsgTkcZu 

CHEMICALS_MIGRATION_QSARr_SMILES <- read_excel("MIGRATION_data_curation/SECTION 4 - add and standardize SMILES/CHEMICALS_MIGRATION_2022-11-14_QSARr_SMILES.xlsx")

CHEMICALS_MIGRATION_QSARr_SMILES <- filter(CHEMICALS_MIGRATION_QSARr_SMILES, !is.na(Canonical_QSARr)) # filter chemicals without QSARr SMILES # Many chemicals excluded as mixtures/UVCBs and metals
CHEMICALS_MIGRATION_QSARr_SMILES <- filter(CHEMICALS_MIGRATION_QSARr_SMILES, is.na(Reason_discarded)) # filter chemicals that contain a metal salt (manual - check) # Excluded ions: Cd, Co, Cr, Cu, Fe, Hg, Ni, Pb, Sb, Se, Tl, Zn, Mn, Al, Li, Mg

CHEMICALS_MIGRATION_QSARr_SMILES <- CHEMICALS_MIGRATION_QSARr_SMILES[,c("test_cas","Canonical_QSARr")]
names(CHEMICALS_MIGRATION_QSARr_SMILES)[2] <- "SMILES"

length(unique(CHEMICALS_MIGRATION_QSARr_SMILES$test_cas)) # 2608
length(unique(CHEMICALS_MIGRATION_QSARr_SMILES$SMILES)) # 2431
## Note: some CAS numbers have the same QSARr SMILES; and could be considered as (highly) comparable. 


# @@@@@@@@@@@@@@
## Section 5.1: Add PFAS information ----
# Not used in this project

# mols <- parse.smiles(CHEMICALS_MIGRATION_QSARr_SMILES$SMILES)

# CHEMICALS_MIGRATION_QSARr_SMILES$CF2R <- matches('[#6;D4](F)(F)', mols) # identifies R-CF2-R or R-CF3 # OECD definition
# CHEMICALS_MIGRATION_QSARr_SMILES$CF2Rx2 <- matches('[#6;D4](F)(F)[#6;D4](F)(F)', mols) # identifies R-CF2-CF2-R or R-CF2-CF3



# @@@@@@@@@@@@@@
## Section 5.2: Add Chemical information to ECOTOX database ----

# Combine QSARr SMILES with ECOTOX database + exclude chemicals without QSARr SMILES
ECOTOX_MIGRATION <- merge(ECOTOX_MIGRATION, CHEMICALS_MIGRATION_QSARr_SMILES, by = "test_cas")

## Test other potential missed 'metal' salts
# table(ECOTOX_MIGRATION$ion1)
# table(ECOTOX_MIGRATION$ion2)
# table(ECOTOX_MIGRATION$ion3)
## Kept included ion: Ni (incorrect, chemicals does not contain Ni); 95772 ([)Zn; incorrect, chemical does not contain Zn) # Kept include salt: iodide





# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Section 6: Prepare ECOTOX_MIGRATION dataset for export ----
ECOTOX_MIGRATION2 <- ECOTOX_MIGRATION[,c("test_cas","CAS","chemical_name","dtxsid","SMILES", 
                                         #"CF2R","CF2Rx2", # PFAS information
                                         "common_name", "latin_name", "kingdom","phylum_division", "subphylum_div", "superclass", "class", "tax_order", "family", "genus",
                                         "obs_duration_mean_standardized_d", 
                                         "conc1_mean_standardized_mgL", "endpoint",
                                         "author", "title","source","publication_year")]

ECOTOX_MIGRATION2$obs_duration_mean_standardized_d <- as.numeric(ECOTOX_MIGRATION2$obs_duration_mean_standardized_d)
ECOTOX_MIGRATION2$conc1_mean_standardized_mgL <- as.numeric(ECOTOX_MIGRATION2$conc1_mean_standardized_mgL)

## Includes: #1. Chemical information, #2. Species information, #3. Duration information, #4. Effect concentration information (LC50), #5. Reference information
## Other interesting information: # "media_type","media_type_comments","exposure_type", "exposure_type_comments","study_type", "test_method", "test_type", "application_type","conc1_type","ncbi_taxid",

SPECIES_MIGRATION <- distinct(ECOTOX_MIGRATION[,c(27:37)], common_name, .keep_all = TRUE)


## Export
write_csv(ECOTOX_MIGRATION2, "MIGRATION_data_curation/ECOTOX_MIGRATION_DATASET_2022-11-15.csv")

length(unique(ECOTOX_MIGRATION2$CAS))
length(unique(ECOTOX_MIGRATION2$common_name))
length(unique(ECOTOX_MIGRATION2$latin_name))


