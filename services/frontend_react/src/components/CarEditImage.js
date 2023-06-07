import React from 'react'
import FilerobotImageEditor, { TABS, TOOLS } from 'react-filerobot-image-editor'

const CarEditImage = ({ imageSrc, closeImageEditor }) => {
  const handleSaveClick = (editedImageObject, designState) => {
    console.log('saved', editedImageObject, designState)
  }
  console.log()
  return (
    <div className="text-center">
      <FilerobotImageEditor
        source={imageSrc}
        onSave={handleSaveClick}
        onClose={closeImageEditor}
        annotationsCommon={{
          fill: '#353839',
        }}
        closeAfterSave={true}
        tabsIds={[TABS.ANNOTATE]} // or {['Adjust', 'Annotate', 'Watermark']}
        defaultTabId={TABS.ANNOTATE} // or 'Annotate'
        defaultToolId={TOOLS.TEXT} // or 'Text'
      />
    </div>
  )
}

export default CarEditImage
