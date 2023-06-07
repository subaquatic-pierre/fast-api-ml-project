# pull official base image
FROM node:alpine

# set work directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY ./package.json /app/

COPY ./package-lock.json /app/

RUN npm install --force

COPY . /app/


# start app
CMD ["npm", "start"]