# Oil Change Dataset

See [dataset_overview.ipynb](dataset_overview.ipynb) for an overview of this dataset.

## Object IDs

Object IDs are used as labels in our segmentation models.

Object ID | Object Name
-|-
`0` | `background`
`1` | `oil_bottle`
`2` | `fluid_bottle`
`3` | `oilfilter`
`4` | `funnel`
`5` | `engine`
`6` | `blue_funnel`
`7` | `tissue_box`
`8` | `drill`
`9` | `cracker_box`
`10` | `spam`

## Object Indices

We train pose interpreter networks on the following five objects. Object indices are used to indicate which of the five objects a particular pose estimate corresponds to.

Object Index | Object Name
-|-
`0` | `oil_bottle`
`1` | `fluid_bottle`
`2` | `funnel`
`3` | `engine`
`4` | `blue_funnel`
