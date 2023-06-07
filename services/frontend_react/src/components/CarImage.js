import React from 'react'

import ReactImageMagnify from 'react-image-magnify'
import InnerImageZoom from 'react-inner-image-zoom'

const CarImage = ({ image }) => {
  return (
    <img src={image} alt="Car" />
    // <div>
    //   <ReactImageMagnify
    //     smallImage={{
    //       alt: 'Wristwatch by Ted Baker London',
    //       isFluidWidth: true,
    //       src: image,
    //     }}
    //     largeImage={{
    //       src: image,
    //       width: 2000,
    //       height: 2000,
    //     }}
    //   />
    // </div>
  )
}

export default CarImage
